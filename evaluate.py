import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import UNetQualityInspector, UNetQualityInspectorV2
from datasets import WindflowUnetDataset

def run_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Determine dynamic feature lists based on user options (Matches train.py/datasets.py)
    feature_list = ["qv", "wind"]
    valid_features = valid_features = ["vws_scalar", "vws_component", "temp", "speed", "warp_error", "qv_shear", "lonlad", "pressure", "divergence"]
    if args.features:
        for feat in args.features:
            if feat in valid_features:
                if feat not in feature_list:
                    feature_list.append(feat)

    max_batches_val = None
    if args.max_batches is not None and str(args.max_batches).lower() != "all":
        max_batches_val = int(args.max_batches)

    # 2. Instantiate the exact same production dataset handler
    eval_dataset = WindflowUnetDataset(
        base_data_dir=args.test_data_dir, 
        target_plev=args.target_plev,
        regression_type=args.regression_type,
        features=feature_list,
        patches_per_batch=args.patches_per_batch,
        max_batches=max_batches_val
    )
    
    # 3. Dynamically catch true input tensor shapes directly from the dataset 
    sample_x, _ = eval_dataset[0]
    computed_in_channels = sample_x.shape[0]
    
    print("=" * 60)
    print(f"RUNNING DATASET-INTEGRATED QUANTITATIVE METRICS EVALUATION")
    print(f"Active Feature Suite  : {feature_list}")
    print(f"Input Tensor Channels : {computed_in_channels} channels")
    print(f"Regression Objective  : {args.regression_type.upper()}")
    print(f"Total Patches Found   : {len(eval_dataset)}")
    print(f"Hardware Compute Node : {device}")
    print("=" * 60)
    
    # 4. Dynamic Architecture Output Determination
    out_channels = 1 if args.regression_type == "magnitude" else 2

    # Instantiate and load model weights dynamically
    if args.model_version == "v1":
        model = UNetQualityInspector(in_channels=computed_in_channels, out_channels=out_channels).to(device)
    elif args.model_version == "v2":
        model = UNetQualityInspectorV2(in_channels=computed_in_channels, out_channels=out_channels).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()
    
    # 5. Configure DataLoader for parallel operational inference batching
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True
    )
    
    all_gt_errors = []
    all_pred_errors = []
    
    # 6. Execution Inference Engine Loop
    print("Processing parallelized batch inference passes...")
    with torch.no_grad():
        for inputs, targets in tqdm(eval_loader, desc="Evaluating", leave=True, mininterval=2.0):
            inputs = inputs.to(device)
            
            # Forward prediction pass through the model
            predictions = model(inputs) 
            
            # Move results safely out of CUDA VRAM to system memory arrays
            all_gt_errors.append(targets.cpu().numpy())
            all_pred_errors.append(predictions.cpu().numpy())

    # Flatten collected blocks into unified 1D arrays for validation math
    y_true_raw = np.concatenate(all_gt_errors, axis=0).flatten()
    y_pred_raw = np.concatenate(all_pred_errors, axis=0).flatten()

    # =========================================================================
    # 7. LATITUDE GRID EXTRACTION & FILTERING MASK
    # =========================================================================
    print("Extracting spatial latitude coordinates for polar masking...")
    all_lats = []
    
    # Iterate through individual patch metadata in dataset order
    for idx in range(len(eval_dataset)):
        batch_idx = idx // eval_dataset.patches_per_batch
        slice_idx = idx % eval_dataset.patches_per_batch
        _, p_target = eval_dataset.all_batches[batch_idx]
        
        with np.load(p_target) as d_target:
            lat_rad = d_target["lat_rad"][slice_idx]
            lon_rad = d_target["lon_rad"][slice_idx]
            _, lat_grid = np.meshgrid(lon_rad, lat_rad)
            lat_deg = np.rad2deg(lat_grid)  # Convert radians to degrees
            
            # Stack across output channels (1 for magnitude, 2 for component)
            if out_channels == 2:
                lat_patch = np.stack([lat_deg, lat_deg], axis=0)
            else:
                lat_patch = np.expand_dims(lat_deg, axis=0)
                
            all_lats.append(lat_patch)
            
    y_lat = np.concatenate(all_lats, axis=0).flatten()

    # Define filtering masks based on advisor criteria
    lat_mask = (y_lat >= -60.0) & (y_lat <= 60.0)
    error_mask = (y_true_raw >= 0.0) & (y_true_raw <= 5.0)
    
    # Combined Boolean Mask
    valid_mask = lat_mask & error_mask
    
    # Apply spatial & magnitude filters
    y_true = y_true_raw[valid_mask]
    y_pred = y_pred_raw[valid_mask]


    # # 7. Global Statistical Evaluation Metrics
    # mae = np.mean(np.abs(y_true - y_pred))
    # rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # ss_res = np.sum((y_true - y_pred) ** 2)
    # ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # r2 = 1 - (ss_res / ss_tot)
    
    # baseline_mae = np.mean(np.abs(y_true))
    # baseline_rmse = np.sqrt(np.mean(y_true ** 2))
    # improvement = ((baseline_mae - mae) / baseline_mae) * 100

    # # 8. Output Evaluation Log Metrics Summary
    # print("\n" + "="*54)
    # print(f"    U-NET PIPELINE QUALITY INSPECTOR REPORT    ")
    # print("="*54)
    # print(f"Evaluation Mode           : {args.regression_type.upper()}")
    # print(f"Target Column Altitude    : {args.target_plev} hPa")
    # print(f"Total Input Channels      : {computed_in_channels}")
    # print(f"Total Pixels Inspected    : {len(y_true):,}")
    # print(f"U-Net Evaluation MAE      : {mae:.4f} m/s")
    # print(f"U-Net Evaluation RMSE     : {rmse:.4f} m/s")
    # print(f"Operational R² Score      : {r2:.4f}")
    # print("-"*54)
    # print(f"Windflow MAE  : {baseline_mae:.4f} m/s")
    # print(f"Windflow RMSE  : {baseline_rmse:.4f} m/s")
    # print(f"U-Net Overall Performance Gain: {improvement:.2f}%")
    # print("="*54 + "\n")

    # =========================================================================
    # 7. METADATA EXTRACTION (LATITUDE, PRESSURE, & TRUE WIND SPEED)
    # =========================================================================
    print("Extracting spatial metadata & ground-truth wind speed...")
    all_lats = []
    all_plevs = []
    all_ws_true = []
    
    for idx in range(len(eval_dataset)):
        batch_idx = idx // eval_dataset.patches_per_batch
        slice_idx = idx % eval_dataset.patches_per_batch
        plev_str, p_target = eval_dataset.all_batches[batch_idx]
        plev_val = float(plev_str)
        
        with np.load(p_target) as d_target:
            lat_rad = d_target["lat_rad"][slice_idx]
            lon_rad = d_target["lon_rad"][slice_idx]
            _, lat_grid = np.meshgrid(lon_rad, lat_rad)
            lat_deg = np.rad2deg(lat_grid)
            
            # Ground truth wind speed calculation
            u_t = d_target["u_lbl_1"][slice_idx]
            v_t = d_target["v_lbl_1"][slice_idx]
            ws_true = np.sqrt(u_t**2 + v_t**2)
            
            if out_channels == 2:
                lat_patch = np.stack([lat_deg, lat_deg], axis=0)
                plev_patch = np.full((2, lat_deg.shape[0], lat_deg.shape[1]), plev_val, dtype=np.float32)
                ws_patch = np.stack([ws_true, ws_true], axis=0)
            else:
                lat_patch = np.expand_dims(lat_deg, axis=0)
                plev_patch = np.full((1, lat_deg.shape[0], lat_deg.shape[1]), plev_val, dtype=np.float32)
                ws_patch = np.expand_dims(ws_true, axis=0)
                
            all_lats.append(lat_patch)
            all_plevs.append(plev_patch)
            all_ws_true.append(ws_patch)
            
    y_lat = np.concatenate(all_lats, axis=0).flatten()
    y_plev = np.concatenate(all_plevs, axis=0).flatten()
    y_ws_true = np.concatenate(all_ws_true, axis=0).flatten()

    # Apply masks
    lat_mask = (y_lat >= -60.0) & (y_lat <= 60.0)
    error_mask = (y_true_raw >= 0.0) & (y_true_raw <= 5.0)
    valid_mask = lat_mask & error_mask
    
    y_true = y_true_raw[valid_mask]
    y_pred = y_pred_raw[valid_mask]
    y_ws_filt = y_ws_true[valid_mask]
    y_plev_filt = y_plev[valid_mask]
    
    # =========================================================================
    # 8b. PER-PRESSURE-LEVEL METRICS BREAKDOWN
    # =========================================================================
    unique_plevs = sorted(np.unique(y_plev_filt), reverse=True) # [700, 600, 500, 400, 300]
    
    plev_list = []
    r2_list = []
    mae_list = []
    rmse_list = []

    print("\n" + "="*68)
    print(f"       PER-PRESSURE-LEVEL PERFORMANCE BREAKDOWN       ")
    print("="*68)
    print(f"{'Pressure (hPa)':<15} | {'Pixels':<12} | {'R² Score':<10} | {'MAE (m/s)':<10} | {'RMSE (m/s)':<10}")
    print("-" * 68)

    for plev in unique_plevs:
        p_mask = (y_plev_filt == plev)
        yt_p = y_true[p_mask]
        yp_p = y_pred[p_mask]
        
        mae_p = np.mean(np.abs(yt_p - yp_p))
        rmse_p = np.sqrt(np.mean((yt_p - yp_p) ** 2))
        
        ss_res_p = np.sum((yt_p - yp_p) ** 2)
        ss_tot_p = np.sum((yt_p - np.mean(yt_p)) ** 2)
        r2_p = 1 - (ss_res_p / ss_tot_p)
        
        plev_list.append(int(plev))
        r2_list.append(r2_p)
        mae_list.append(mae_p)
        rmse_list.append(rmse_p)
        
        print(f"{int(plev):<15} | {len(yt_p):<12,} | {r2_p:<10.4f} | {mae_p:<10.4f} | {rmse_p:<10.4f}")
    print("="*68 + "\n")


    os.makedirs("plots", exist_ok=True)
    print("Generating 2D Density Plot for R² analysis... (this might take a minute with large datasets)")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the Hexbin Density Plot
    hb = ax.hexbin(y_true, y_pred, gridsize=150, cmap='turbo', mincnt=1)

    # Add the "Perfect Prediction" Diagonal Line (y = x)
    # min_val = min(np.min(y_true), np.min(y_pred))
    min_val = 0.0
    # max_val = max(np.max(y_true), np.max(y_pred))
    max_val = 5.0
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')

    # Add the Colorbar
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of Pixels per Bin', fontsize=12)

    # Add Text Annotations (R² and MAE)
    text_str = f"$R^2$ = {r2:.4f}\nMAE = {mae:.4f} m/s"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Format the Axes and Titles
    ax.set_xlim([0.0, 5.0])
    ax.set_ylim([0.0, 5.0])
    ax.set_title(f'U-Net Regression Performance ({args.target_plev} hPa)', fontsize=16)
    ax.set_xlabel('Ground Truth RAFT Error (m/s)', fontsize=14)
    ax.set_ylabel('U-Net Predicted Error (m/s)', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower right')

    # Save the figure out to your plots folder
    plot_filename = f"{args.plot_name}.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()

    print(f"Plot successfully saved to: {plot_filename}\n")

    # =========================================================================
    # 9. VERTICAL PROFILE PLOT GENERATION (R² vs Pressure Level)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(6, 8))

    ax.plot(r2_list, plev_list, marker='o', linewidth=2.5, markersize=8, color='#0284C7', label='$R^2$ Score')

    # Standard Atmospheric Orientation: Invert Y-axis so 700 hPa is bottom, 300 hPa is top
    ax.invert_yaxis()
    
    ax.set_title('Vertical Profile of Error Inspector $R^2$', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Operational $R^2$ Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pressure Level (hPa)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate values on the plot
    for r2_val, plev_val in zip(r2_list, plev_list):
        ax.annotate(f' {r2_val:.3f}', (r2_val, plev_val), textcoords="offset points", xytext=(5, -4), fontsize=10, fontweight='bold')

    profile_plot_filename = f"{args.plot_name}_vertical_profile.png"
    plt.tight_layout()
    plt.savefig(profile_plot_filename, dpi=300)
    plt.close()

    print(f"Vertical profile plot saved to: {profile_plot_filename}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Metrics Evaluation Engine for U-Net Quality Inspector")
    
    # Execution Toggle Parameters
    parser.add_argument(
        "--features", 
        type=str, 
        nargs="+", 
        default=["qv", "wind"],
        help="List of extra features to add (Must match configurations used during training)"
    )
    parser.add_argument("--model_version", type=str, default="v1", choices=["v1", "v2"],
                        help="Version of the U-Net model architecture to evaluate: 'v1' or 'v2'")
    parser.add_argument("--regression_type", type=str, default="magnitude", choices=["magnitude", "component"],
                        help="Target output mapping configuration evaluated: 'magnitude' or 'component'")
    parser.add_argument("--target_plev", type=str, default="all",
                        help="Target pressure level for evaluation")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Input batch size for parallel inference loading")
    parser.add_argument("--max_batches", default=None, type=str, 
                        help="Number of batch files to evaluate with (Optional, evaluates all by default)")
    parser.add_argument("--patches_per_batch", default=20, required=False, type=int,
                        help="Number of patches to sample per batch (Optional, default=20)")                    
    # File Management Paths
    parser.add_argument("--test_data_dir", type=str, 
                        default="/ships22/cryo/daves/windflow/ssung/windflow_test_outputs/",
                        help="Path to folder containing target folders or root pressure directories")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to the targeted model checkpoint weights file")
    parser.add_argument("--plot_name", type=str, default="plots/plot")

    args = parser.parse_args()
    run_evaluate(args)
