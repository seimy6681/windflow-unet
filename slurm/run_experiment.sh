#!/bin/bash
#SBATCH --job-name=unet_matrix
#SBATCH -p salvador   
#SBATCH --nodelist=gustav
#SBATCH --output=logs/unet_%A_%a.out  # %A is the master job ID, %a is the array ID
#SBATCH --error=logs/unet_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # Matches the num_workers=4 in your DataLoader
#SBATCH --gres=gpu:1                  # Request 1 GPU per task
#SBATCH --mem=32G                     # Adjust memory as needed
#SBATCH --time=3-00:01:00               # 12 hours max per job
#SBATCH --array=0-3                   # This creates 4 identical parallel jobs!

# Load your environments here (e.g., module load conda, conda activate myenv)
source ~/.bashrc
conda activate pytorch_windflow

# ---------------------------------------------------------------------
# GLOBAL EXPERIMENT PARAMETERS
# ---------------------------------------------------------------------
TARGET_PLEV=400

# ---------------------------------------------------------------------
# DYNAMIC EXPERIMENT ROUTING
# ---------------------------------------------------------------------
# SLURM_ARRAY_TASK_ID will be 0, 1, 2, or 3 depending on the parallel instance.
# We use a simple case switch to assign the correct arguments to each instance.

case $SLURM_ARRAY_TASK_ID in
    0)
        CHANNELS=3
        REG="magnitude"
        EXTRA_FLAGS=""
        ;;
    1)
        CHANNELS=3
        REG="component"
        EXTRA_FLAGS="--max_batches 1 --feature speed --model_save_path '400_hpa_batch_0_speed_${CHANNELS}ch_${REG}'"
        ;;
    2)
        CHANNELS=4
        REG="magnitude"
        EXTRA_FLAGS="--feature vws"
        ;;
    3)
        CHANNELS=4
        REG="component"
        EXTRA_FLAGS="--max_batches 1 --feature vws_component --model_save_path '400_hpa_${CHANNELS}ch_${REG}'"
        ;;
esac

echo "================================================================="
echo "U-Net Training starting ..."
echo "Starting Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
echo "Configuration  : ${CHANNELS}-Channel | ${REG} ${EXTRA_FLAGS}"
echo "================================================================="

# Execute the python script with the dynamically assigned variables
python unet/train.py \
    --channels $CHANNELS \
    --regression_type $REG \
    --target_plev $TARGET_PLEV \
    $EXTRA_FLAGS