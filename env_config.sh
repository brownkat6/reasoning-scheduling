#!/bin/bash
# Environment configuration for reasoning-scheduling project
# Source this file to set up your environment: source env_config.sh

# Set default Python command (override by setting PYTHON_CMD environment variable)
export PYTHON_CMD=${PYTHON_CMD:-python3}

# Set default data directory (override by setting DATA_DIR environment variable)
export DATA_DIR=${DATA_DIR:-./data}

# Set default output directory (override by setting OUTPUT_DIR environment variable)
export OUTPUT_DIR=${OUTPUT_DIR:-./outputs}

# Common cluster-specific configurations
# Uncomment and modify as needed for your cluster

# Harvard cluster (Katrina)
# export PYTHON_CMD="/n/netscratch/dwork_lab/Lab/katrina/envs/reasoning/bin/python"
# export DATA_DIR="/n/netscratch/gershman_lab/Lab/amuppidi/reasoning_scheduling_new_orig/data"
# export OUTPUT_DIR="/n/netscratch/gershman_lab/Lab/amuppidi/reasoning"

# Harvard cluster (Amuppidi)  
# export PYTHON_CMD="~/.conda/envs/torch/bin/python"
# export DATA_DIR="/n/netscratch/gershman_lab/Lab/amuppidi/reasoning_scheduling_new_orig/data"
# export OUTPUT_DIR="/n/netscratch/gershman_lab/Lab/amuppidi/reasoning"

# Local development
# export PYTHON_CMD="python3"
# export DATA_DIR="./data"
# export OUTPUT_DIR="./outputs"

echo "Environment configured:"
echo "  PYTHON_CMD: $PYTHON_CMD"
echo "  DATA_DIR: $DATA_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"