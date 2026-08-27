import os
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
from utils import calculate_photometric_residual

# helper
def resolve_file_path(base_dir, plev_str, batch_filename):
    """Checks test directory first; falls back to train directory if missing."""
    primary_path = os.path.join(base_dir, plev_str, batch_filename)
    if os.path.exists(primary_path):
        return primary_path
    
    # Switch folder from test <-> train as fallback
    if "windflow_test_outputs" in base_dir:
        fallback_dir = base_dir.replace("windflow_test_outputs", "windflow_outputs")
    else:
        fallback_dir = base_dir.replace("windflow_outputs", "windflow_test_outputs")
        
    fallback_path = os.path.join(fallback_dir, plev_str, batch_filename)
    if os.path.exists(fallback_path):
        return fallback_path


class WindflowUnetDataset(Dataset):
    """
    A single, scalable dataset handling 3-channel, 4-channel, vector-shear, 
    and component regression dynamically. Avoids redundant classes.
    """
    def __init__(self, base_data_dir, target_plev, regression_type="component", features=["qv", "wind", "vws_component"], patches_per_batch=100, max_batches=None):
        """
        Args:
            base_data_dir (str): Root path containing level sub-folders ('700', '600', etc.)
            target_plev (float or str): Target pressure level (e.g., 700)
            regression_type (str): "magnitude" or "component"
            features (list): Ordered list of features to compile. 
                             Options: ["qv", "wind", "vws_scalar", "vws_component"]
            patches_per_batch (int): Cached count of array sequences inside one .npz batch file
        """
        if regression_type not in ["magnitude", "component"]:
            raise ValueError("regression_type must be 'magnitude' or 'component'")
            
        self.base_data_dir = base_data_dir
        self.patches_per_batch = patches_per_batch
        self.regression_type = regression_type
        self.features = features


        self.pressure_levels = np.array([300.0, 400.0, 500.0, 600.0, 700.0], dtype=np.float64) # TEMPORARY


        # self.pressure_levels = np.array([200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 850.0], dtype=np.float64)
        self.target_plev = target_plev
        
        # 1. Determine which levels to load without crashing on "all"
        self.all_batches = []
        if str(target_plev).lower() == "all":
            levels_to_load = self.pressure_levels
        else:
            levels_to_load = [float(target_plev)]    
        
        # 2. Dynamic File Discovery Loop
        for plev in levels_to_load:
            plev_str = str(int(plev))
            plev_dir = os.path.join(self.base_data_dir, plev_str)
            
            if os.path.exists(plev_dir):
                # Temporarily hold the files for this specific level
                batch_files = sorted(glob.glob(os.path.join(plev_dir, "*.npz")))
                
                # Apply the max_batches limit if one was provided
                if max_batches is not None:
                    batch_files = batch_files[:max_batches]
                    
                # Add them to the master global list
                for bf in batch_files:
                    self.all_batches.append((plev_str, bf))

        # 3. Calculate final dataset scope limits
        self.num_files = len(self.all_batches)
        if self.num_files == 0:
            raise FileNotFoundError(f"No .npz files found in target scope: {self.base_data_dir}")

        self.total_patches = self.patches_per_batch * self.num_files
        print("Actual total number of patches: ", self.total_patches)

    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, global_idx):
        batch_idx = global_idx // self.patches_per_batch
        slice_idx = global_idx % self.patches_per_batch

        # 1. Unpack the target level and file path from your master list
        target_str, p_target = self.all_batches[batch_idx]
        target_plev = float(target_str)

        d_target = np.load(p_target)
        
        # 2. Dynamically determine upper/lower bounds for THIS SPECIFIC patch
        d_upper, d_lower = None, None
        if any(f in self.features for f in ["vws_scalar", "vws_component", "qv_shear"]):
            
            # Find where we are in the pressure list
            target_idx = np.where(self.pressure_levels == target_plev)[0][0]
            
            if target_idx == 0:
                upper_str = str(int(self.pressure_levels[0]))
                lower_str = str(int(self.pressure_levels[1]))
            elif target_idx == len(self.pressure_levels) - 1:
                upper_str = str(int(self.pressure_levels[-2]))
                lower_str = str(int(self.pressure_levels[-1]))
            else:
                upper_str = str(int(self.pressure_levels[target_idx - 1]))
                lower_str = str(int(self.pressure_levels[target_idx + 1]))

            # Grab just the filename (e.g., 'batch_5.npz') so we can find it in the other folders
            batch_filename = os.path.basename(p_target)
            
            # p_upper = os.path.join(self.base_data_dir, upper_str, batch_filename)
            # p_lower = os.path.join(self.base_data_dir, lower_str, batch_filename)
            
            # smart lookup 
            p_upper = resolve_file_path(self.base_data_dir, upper_str, batch_filename)
            p_lower = resolve_file_path(self.base_data_dir, lower_str, batch_filename)
            
            d_upper = np.load(p_upper)
            d_lower = np.load(p_lower)

            
        x_channels = []

        # 2. Dynamic Input Construction Loop
        for feat in self.features:
            # Create a temporary list for this specific feature's channels
            feat_channels = []

            if feat == "qv":
                feat_channels.append(d_target["qv_t0"][slice_idx].squeeze())
                
            elif feat == "wind":
                feat_channels.append(d_target["u_pred"][slice_idx])
                feat_channels.append(d_target["v_pred"][slice_idx])

            elif feat == "speed":
                u_p = d_target["u_pred"][slice_idx]
                v_p = d_target["v_pred"][slice_idx]
                feat_channels.append(np.sqrt(u_p**2 + v_p**2))
                
            elif feat == "vws_scalar":
                u_shear = d_upper["u_pred"][slice_idx] - d_lower["u_pred"][slice_idx]
                v_shear = d_upper["v_pred"][slice_idx] - d_lower["v_pred"][slice_idx]
                feat_channels.append(np.sqrt(u_shear**2 + v_shear**2))
                
            elif feat == "vws_component":
                u_shear = d_upper["u_pred"][slice_idx] - d_lower["u_pred"][slice_idx]
                v_shear = d_upper["v_pred"][slice_idx] - d_lower["v_pred"][slice_idx]
                feat_channels.append(u_shear)
                feat_channels.append(v_shear)
            
            elif feat == "qv_shear":
                qv_shear = d_upper["qv_t0"][slice_idx].squeeze() - d_lower["qv_t0"][slice_idx].squeeze()
                feat_channels.append(qv_shear)

            elif feat == "warp_error":
                qv_0 = d_target["qv_t0"][slice_idx].squeeze()
                qv_1 = d_target["qv_t1"][slice_idx].squeeze() 
                u_p = d_target["u_pred"][slice_idx]
                v_p = d_target["v_pred"][slice_idx]
                dt_seconds = 1800.0 
                dx_meters = 7000.0 
                u_pixels = (u_p * dt_seconds) / dx_meters 
                v_pixels = (v_p * dt_seconds) / dx_meters
                feat_channels.append(calculate_photometric_residual(qv_0, qv_1, u_pixels, v_pixels))

            elif feat == "lonlad":
                lat_rad = d_target["lat_rad"][slice_idx]
                lon_rad = d_target["lon_rad"][slice_idx]
                
                # Create the 2D spatial coordinate meshes
                lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)
                
                # Scale both cleanly between -1.0 and 1.0
                feat_channels.append(lat_grid / (np.pi / 2))  # Latitude normalized
                feat_channels.append(lon_grid / np.pi)
                
            elif feat == "pressure":
                p_norm = target_plev / 1000.0
                grid_shape = d_target["qv_t0"][slice_idx].squeeze().shape
                feat_channels.append(np.full(grid_shape, p_norm, dtype=np.float32))

            elif feat == "divergence":
                u_p = d_target["u_pred"][slice_idx]
                v_p = d_target["v_pred"][slice_idx]

                dx_meters = 7000.0
                dy_meters = 7000.0

                du_dx = np.gradient(u_p, dx_meters, axis=-1)
                dv_dy = np.gradient(v_p, dy_meters, axis=-2)
                feat_channels.append(du_dx + dv_dy)

            # Z-score scale the channel(s) right now IF it's a raw physical feature
            if feat not in ["lonlad", "pressure"]:
                for i in range(len(feat_channels)):
                    mean_val = np.mean(feat_channels[i])
                    std_val = np.std(feat_channels[i])
                    feat_channels[i] = (feat_channels[i] - mean_val) / (std_val + 1e-8)

            # Dump them into the final master array
            x_channels.extend(feat_channels)

        # Merge them altogether into the final PyTorch tensor structure
        x_input = np.stack(x_channels, axis=0)
        
        # 3. Label Output Routing Extraction
        u_true, v_true = d_target["u_lbl_1"][slice_idx], d_target["v_lbl_1"][slice_idx]
        u_pred, v_pred = d_target["u_pred"][slice_idx], d_target["v_pred"][slice_idx]

        if self.regression_type == "magnitude":
            total_error = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
            y_target = np.expand_dims(total_error, axis=0)
        elif self.regression_type == "component":
            # y_target = np.stack([np.abs(u_pred - u_true), np.abs(v_pred - v_true)], axis=0)
            y_target = np.stack([np.abs(u_pred - u_true), np.abs(v_pred - v_true)], axis=0)

        return (
            torch.tensor(x_input, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32)
        )
    

