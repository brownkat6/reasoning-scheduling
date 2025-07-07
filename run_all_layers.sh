mkdir -p logs

# Loop through all layers from 0 to 28
for layer in {0..28}; do
    echo "Submitting job for layer $layer"
    # Call generate_data.sh with True for X data, False for Y data, and the specific layer
    # Get script directory for relative path
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"    
    sbatch "$SCRIPT_DIR/generate_data.sh" True False $layer
    sleep 0.3
done

echo "All layer jobs submitted"