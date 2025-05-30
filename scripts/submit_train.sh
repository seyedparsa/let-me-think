#!/bin/bash
#SBATCH --job-name=reason              # Name of your job
#SBATCH --output=logs/%x-%j.out        # Output file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x-%j.err         # Error file
#SBATCH --time=04:00:00                # Maximum run time (hh:mm:ss)
#SBATCH --partition=gpuA100x4           # Specify the GPU partition
#SBATCH --cpus-per-task=1              # Number of CPU cores
#SBATCH --mem=16G                      # Memory needed (adjust as required)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --gpus=1                       # Number of GPUs
#SBATCH --account=bbjr-delta-gpu       # (Optional) Specify your account if needed

# Run the training script using accelerate
# accelerate launch --config_file configs/accelerate.yaml train.py --wandb --config configs/training_config.yaml
# conda activate nueva
echo "Starting wandb agent with SWEEP_ID: $SWEEP_ID"
source ~/.bashrc
conda activate nueva
wandb agent "seyedparsa/bepatient/$SWEEP_ID" --count 1