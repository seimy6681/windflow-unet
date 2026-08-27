import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np
from tqdm import tqdm
import wandb

from models import UNetQualityInspector, UNetQualityInspectorV2
# Import your clean production dataset handler
from datasets import WindflowUnetDataset
from evaluate import run_evaluate
import random

def set_global_seed(seed):
    """Locks down all random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Force cuDNN to be deterministic (might slow down training slightly but guarantees exact replication)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================================
# CORE TRAINING LOOP
# =====================================================================
def run_training(args):
    set_global_seed(args.seed)
    # Check for hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Determine dynamic feature lists based on user options
    feature_list = ["qv", "wind"]

    # Valid options you want to allow to prevent typos
    valid_features = [
                    "vws_scalar", 
                    "vws_component", 
                    "temp", 
                    "speed", 
                    "warp_error",
                    "qv_shear",
                    "lonlad",
                    "pressure",
                    "divergence"
                    ]
    
    # 2. Dynamically add user-requested features
    if args.features:  # checks if the user provided any extra features
        for feat in args.features:
            if feat == "wind":
                continue
            if feat == "qv":
                continue
            if feat in valid_features:
                if feat not in feature_list: # Prevent accidental duplicates
                    feature_list.append(feat)
            else:
                print(f"Warning: '{feat}' is not a recognized feature. Skipping.")
    
    print(f"Final feature list for U-Net: {feature_list}")
        
    print("=" * 60)
    print(f"LAUNCHING UNIFIED U-NET QUALITY INSPECTOR")
    print(f"Target Computing Node : {device}")
    print(f"Regression Objective  : {args.regression_type.upper()}")
    print(f"Active Feature Suite  : {feature_list}")
    print("=" * 60)

    # Smart naming engine for stacked flags (e.g., "vws_component_temp_component_run")
    feature_suffix = "_".join(args.features) if args.features else "3ch"
    run_name = f"{feature_suffix}_{args.regression_type}_run"

    wandb.init(
        project="unet-windflow-inspector", 
        config=vars(args),                  
        name=run_name 
    )

    # Instantiate the unified dynamic dataset
    full_dataset = WindflowUnetDataset(
        base_data_dir=args.data_dir, 
        target_plev=args.target_plev,
        regression_type=args.regression_type,
        features=feature_list,
        patches_per_batch=100,
        max_batches=args.max_batches
    )
    
    # Peek at one sample to dynamically catch true input tensor shapes
    sample_x, _ = full_dataset[0]
    computed_in_channels = sample_x.shape[0]
    
    total_samples = len(full_dataset)

    num_plevs = 5  # 300, 400, 500, 600, 700
    samples_per_plev = total_samples // num_plevs

    # 1. Create a matching list of dummy labels for stratification
    # E.g., [0,0,0... 1,1,1... 2,2,2...] so scikit-learn knows which plev each index belongs to
    plev_labels = np.repeat(np.arange(num_plevs), samples_per_plev)
    indices = np.arange(total_samples)

    # 2. Use scikit-learn to do a perfectly stratified split on the indices
    train_indices, val_indices = train_test_split(
        indices,
        test_size=args.val_split,
        stratify=plev_labels,   # <--- THE MAGIC LINE: Guarantees equal plev distribution
        random_state=42         # Equivalent to manual_seed(42)
    )

    # 3. Wrap them back into PyTorch Subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Sklearn Stratified Split Complete!")
    print(f"Total Train Patches: {len(train_dataset)}")
    print(f"Total Val Patches:   {len(val_dataset)}")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Configure DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("=" * 60)
    print("                 U-NET PIPELINE INITIALIZATION                 ")
    print("=" * 60)
    if args.max_batches is not None:
        print(f"Training Scope        : Capped at {args.max_batches} batch file(s)")
    else:
        print(f"Training Scope        : Full Dataset (All available batches)")
    
    print(f"Total Patches Found   : {len(full_dataset)}")
    print(f"Input Tensor Channels : {computed_in_channels} channels")
    print(f"Training Allocation   : {len(train_dataset)} patches ({100 - int(args.val_split*100)}%)")
    print(f"Validation Allocation : {len(val_dataset)} patches ({int(args.val_split*100)}%)")
    print(f"Learning Rate         : {args.lr}")
    print("=" * 60)

    # Determine Model Output Channels
    out_channels = 1 if args.regression_type == "magnitude" else 2
        
    if args.model_version == "v1":
        # Inject dynamically calculated in_channels directly into model setup
        model = UNetQualityInspector(in_channels=computed_in_channels, out_channels=out_channels).to(device)
    elif args.model_version == "v2":
        model = UNetQualityInspectorV2(in_channels=computed_in_channels, out_channels=out_channels).to(device)
    
    wandb.watch(model, log="all", log_freq=10)

    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    global_step = 0  # Fixed scope tracking variable placement
    log_interval = 10

    # Main Epoch Iteration
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        running_step_loss = 0.0
        
        # Wrapped loop in tqdm configured to protect Slurm passive output streams
        train_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1:02d}/{args.epochs} [Train]", 
            leave=True,
            mininterval=2.0
        )
        
        for batch_idx, (inputs, targets) in enumerate(train_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            train_loss += current_loss * inputs.size(0)
            running_step_loss += current_loss
            global_step += 1

            train_bar.set_postfix({"loss": f"{current_loss:.4f}"})

            # --- Step-Level Logging ---
            if (batch_idx + 1) % log_interval == 0:
                avg_step_loss = running_step_loss / log_interval
                
                wandb.log({
                    "train/step_loss": avg_step_loss,
                    "global_step": global_step
                })
                
                train_bar.set_description(f"Epoch {epoch+1:02d} | Step Loss: {avg_step_loss:.4f}")
                running_step_loss = 0.0
            
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d} [Val]", leave=False, mininterval=5.0)
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * inputs.size(0)
                
        # Metric Normalization
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        
        print(f"\nSummary | Epoch [{epoch+1:02d}] Train Loss: {epoch_train_loss:.4f} m/s | Val Loss: {epoch_val_loss:.4f} m/s")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": epoch_train_loss,
            "val/loss": epoch_val_loss,
            "best_val_loss": min(epoch_val_loss, best_val_loss)
        })

        # Checkpoint Saving Logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f" ---> Checkpoint Saved! (New lowest validation error discovered)")

    print("=" * 60)
    print("TRAINING CYCLE COMPLETE")
    print(f"Optimal model parameters exported to: '{args.model_save_path}'")
    print("=" * 60)
    
    wandb.finish()

    # -----------------------------------------------------------------
    # AUTOMATED POST-TRAINING EVALUATION
    # -----------------------------------------------------------------
    print("\n" + "*" * 60)
    print("INITIATING AUTOMATED POST-TRAINING EVALUATION")
    print("*" * 60)
    
    args.weights_path = args.model_save_path
    run_evaluate(args)


# =====================================================================
# 3. COMMAND LINE INTERFACE (CLI)
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unified U-Net Quality Inspector for Windflow Outputs")
    parser.add_argument("--seed", default=42, type=int,
                    help="Global random seed for reproducibility")

    parser.add_argument("--model_version", default="v1", choices=["v1", "v2"], type=str,
                        help="Version of the U-Net model architecture to train: 'v1' or 'v2'")
    parser.add_argument(
        "--features", 
        type=str, 
        nargs="+",             # "+" tells argparse to accept 1 or more space-separated strings into a list
        default=["qv", "wind"], # Optional: sets the baseline features if the user passes nothing
        help="List of extra features to add (e.g., --features speed vws_component warp_error)"
    )
    # Paths
    parser.add_argument("--data_dir", default="/ships22/cryo/daves/windflow/ssung/windflow_outputs/", type=str,
                        help="Path to the directory containing batch_*.npz files")
    parser.add_argument("--test_data_dir", default="/ships22/cryo/daves/windflow/ssung/windflow_test_outputs/", type=str,
                        help="Path to the test data directory.")
    parser.add_argument("--model_save_path", default=None, type=str,
                        help="Filename destination for exporting weights. Generated dynamically if left as None.")
    
    # Hyperparameters
    parser.add_argument("--target_plev", default="all", type=str,
                        help="Target pressure level for training (e.g., '700' or 'all')")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Input batch size for training")
    parser.add_argument("--epochs", default=30, type=int,
                        help="Number of full training passes over the dataset")
    parser.add_argument("--lr", "--learning_rate", default=2e-4, type=float,
                        help="Learning rate for the Adam optimizer")
    parser.add_argument("--val_split", default=0.2, type=float,
                        help="Fraction of dataset reserved for validation tracking")
    parser.add_argument("--regression_type", default="component", choices=["magnitude", "component"], type=str,
                        help="Type of regression target: 'magnitude' or 'component'")
    parser.add_argument("--max_batches", default=None, type=int, help="Number of batches to train with (100 patches per batch)")
    parser.add_argument("--patches_per_batch", default=20, type=int, help="Number of test patches per hPa to evaluate on")
    args = parser.parse_args()
    
    # Smart output naming engine matching user configurations automatically
    feature_suffix = "_".join(args.features) if args.features else "3ch"
    if args.model_save_path is None:
        os.makedirs("weights", exist_ok=True)
        args.model_save_path = f"weights/best_{feature_suffix}_{args.regression_type}_unet_quality_inspector.pth"
    
    # Fire off execution
    run_training(args)