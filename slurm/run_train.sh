#!/bin/bash
#SBATCH --job-name=unet_matrix
#SBATCH -p salvador   
#SBATCH --output=logs/tmp_%A_%a.out  # Temporary log, renamed automatically at completion
#SBATCH --error=logs/tmp_%a.err        
#SBATCH --gpus=1                  
#SBATCH --mem=32G                     
#SBATCH -t 12:00:00             
#SBATCH --array=0-3                   

# Load your environments
source ~/.bashrc
conda activate pytorch_windflow

# ---------------------------------------------------------------------
# GLOBAL EXPERIMENT PARAMETERS
# ---------------------------------------------------------------------
TARGET_PLEV=400

# ---------------------------------------------------------------------
# DYNAMIC EXPERIMENT ROUTING (Just pick your features and limits!)
# ---------------------------------------------------------------------
case $SLURM_ARRAY_TASK_ID in
    0)
        REG="component"
        MAX_BATCHES="1"
        FEATURES="qv wind"
        ;;
    1)
        REG="component"
        MAX_BATCHES="1"
        FEATURES="qv wind speed"
        ;;
    2)
        REG="component"
        MAX_BATCHES="1"
        FEATURES="qv wind vws_component"
        ;;
    3)
        REG="component"
        MAX_BATCHES="1"
        FEATURES="qv wind warp_error"
        ;;
esac

# ---------------------------------------------------------------------
# AUTOMATED CHANNEL COUNTING & STRING FORMATTING
# ---------------------------------------------------------------------

# 1. Count the channels dynamically! 
# Converting the string to a bash array handles counting instantly.
FEAT_ARRAY=($FEATURES)
CHANNELS=${#FEAT_ARRAY[@]}+1

# 2. Format the features string for the filename
FEAT_STR=$(echo $FEATURES | tr ' ' '_')

# 3. Format the batch string and execution flag
if [ "$MAX_BATCHES" = "all" ]; then
    BATCH_FLAG=""
    BATCH_STR="all_batches"
else
    BATCH_FLAG="--max_batches $MAX_BATCHES"
    BATCH_STR="${MAX_BATCHES}_batches"
fi

# 4. Construct the definitive run name
RUN_NAME="${TARGET_PLEV}hPa_${CHANNELS}ch_${FEAT_STR}_${BATCH_STR}_${REG}"
SAVE_PATH="models/${RUN_NAME}.pt"
FINAL_LOG="logs/${RUN_NAME}.log"

# ---------------------------------------------------------------------
# THE AUTOMATIC LOG-RENAMER TRAP
# ---------------------------------------------------------------------
# This function automatically triggers when the script exits, safely 
# renaming SLURM's temp output into your beautiful custom log file.
cleanup_and_rename() {
    if [ -f "logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" ]; then
        mv "logs/tmp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$FINAL_LOG"
    fi
    # Clean up the empty error file if no errors happened
    if [ ! -s "logs/tmp_${SLURM_ARRAY_TASK_ID}.err" ]; then
        rm "logs/tmp_${SLURM_ARRAY_TASK_ID}.err"
    fi
}
trap cleanup_and_rename EXIT

echo "================================================================="
echo "U-Net Training starting ..."
echo "Starting Job ID: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"
echo "Run Name       : $RUN_NAME"
echo "Dynamic Channels Calculated: $CHANNELS"
echo "Final Log will be saved as : $FINAL_LOG"
echo "================================================================="

# Execute the python script with the calculated channels count
python unet/train.py \
    --regression_type $REG \
    --target_plev $TARGET_PLEV \
    --features $FEATURES \
    --model_save_path "$SAVE_PATH" \
    $BATCH_FLAG