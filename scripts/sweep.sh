SWEEP_ID=$(wandb sweep configs/sweep.yaml 2>&1 | grep "Creating sweep with ID:" | awk '{print $NF}')
if [ -z "$SWEEP_ID" ]; then
    echo "Error: Could not extract SWEEP_ID."
    exit 1
fi
NUM_AGENTS=$1
for i in $(seq 1 $NUM_AGENTS); do
    sbatch --export=SWEEP_ID=$SWEEP_ID scripts/submit_train.sh
done
echo "Running sweep with ID: $SWEEP_ID with $NUM_AGENTS agents"
