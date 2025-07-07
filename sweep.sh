#!/bin/bash
# Create a sweep through all layers 0-27 and submit jobs for each

# Base directory for X data - use environment variable or default
BASE_X_STEM="${DATA_DIR:-./data}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Loop through all layers from 0 to 27
for i in {0..27}; do
    # Construct the X_STEM path
    X_STEM="${BASE_X_STEM}/layer_${i}"
    LAYER_NAME="layer_${i}"
    
    echo "Submitting job for layer_${i}"
    echo "X_STEM: $X_STEM"
    
    # Submit the job with appropriate X_STEM
    # Get script directory for relative path
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"    
    sbatch "$SCRIPT_DIR/train.sh" "$X_STEM" "$LAYER_NAME"
    
    # Optional: Add a small delay to avoid overwhelming the scheduler
    sleep 0.5
done

echo "All layer jobs submitted"