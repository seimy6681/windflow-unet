# Windflow U-Net Quality Inspector

A Physics-Informed Neural Network (PINN) framework for atmospheric wind vector error prediction using GOES-18/19 satellite observations and numerical model fields. The pipeline enforces fluid dynamics constraints via horizontal divergence regularization to preserve spatial continuity across multi-level pressure surfaces ($700\text{--}300\text{ hPa}$).

---

## Directory Structure

```text
windflow-unet/
├── unet/
│   ├── models.py       # U-Net architectures (UNetQualityInspector v1 & v2)
│   ├── datasets.py     # PyTorch dataset handler for multi-channel .npz patches
│   ├── losses.py       # Custom physics loss (HorizontalDivergenceLoss regularizer)
│   ├── train.py        # Supervised & physics-informed model training loop
│   └── evaluate.py     # Quantitative evaluation, per-level metrics, and plotting
├── slurm_scripts/
│   ├── run_train.sh    # SLURM array script for training hyperparameter matrices
│   └── run_evaluate.sh # SLURM array script for multi-seed inference evaluation
├── models/             # Saved PyTorch checkpoint weights (.pt)
├── plots/              # Output hexbin density scatters & vertical profile plots
└── logs/               # Automated run logs and execution outputs
```

---

## Regression Objectives

The U-Net model can be trained for two distinct target error representations:

* **`component` (Default, 2 Output Channels):** Predicts absolute errors for zonal ($u$) and meridional ($v$) wind speed components independently.
* **`magnitude` (1 Output Channel):** Predicts the total scalar Euclidean magnitude distance error ($\sqrt{\Delta u^2 + \Delta v^2}$).

---

## Core Modules

* **`unet/models.py`**: Defines `UNetQualityInspector` (V1 standard U-Net) and `UNetQualityInspectorV2` (residual/expanded bottleneck U-Net). Automatically scales input channels based on selected atmospheric features.
* **`unet/datasets.py`**: `WindflowUnetDataset` handles lazy loading and dynamic sampling of 2D atmospheric patch pairs from `.npz` archives. Supports dynamic feature combinations:
  * Moisture & Wind: `qv`, `wind`
  * Atmospheric Dynamics: `pressure`, `lonlad`, `speed`, `temp`
  * Physics Derivatives: `vws_component`, `vws_scalar`, `qv_shear`, `warp_error`, `divergence`
* **`unet/losses.py`**: Implements **`HorizontalDivergenceLoss`**, applying finite-difference spatial derivative kernels ($K_x, K_y$) during backpropagation to penalize non-physical flow expansion ($\mathcal{D} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$).
* **`unet/train.py`**: The central training loop handling GPU device allocation, dynamic feature selection, data loading, backpropagation, and checkpoint saving.
* **`unet/evaluate.py`**: Quantitative diagnostic engine. Computes global and per-pressure-level ($700\text{--}300\text{ hPa}$) metrics ($R^2$, MAE, RMSE, % improvement over baseline Windflow), applies polar coordinate masking ($\pm 60^\circ$ latitude), and generates performance scatter plots and vertical profiles.

---

## Execution Workflow (SLURM Cluster)

### 1. Model Training Matrix
Launch batch training tasks across hyperparameter/feature sets:

```bash
mkdir -p logs models plots
sbatch slurm_scripts/run_train.sh
```
* Configures target pressure levels, regression objectives (`component`/`magnitude`), and feature suites dynamically per array task ID (`SLURM_ARRAY_TASK_ID`).
* Automatically names checkpoint files into `models/` and formats execution output logs.

### 2. Multi-Seed Evaluation Matrix
Run automated evaluation across trained checkpoints using multiple random seeds:

```bash
sbatch slurm_scripts/run_evaluate.sh
```
* Maps array tasks across multiple experiment configurations and 5 random seed iterations.
* Generates per-pressure level metric summaries, hexbin scatter density plots, and vertical profile PNGs into `plots/`.

### 3. Monitoring Cluster Jobs

```bash
# Check queue status for active user
squeue -u $USER

# Tail execution output of an active array job
tail -f logs/eval_400hPa_3ch_qv_wind_1_batches_component_v2_seed42.log
```