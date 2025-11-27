#!/bin/bash

# Sync files FROM HPC to local machine using rsync
# Useful for retrieving results, trained models, or updated notebooks
# Note: The trailing slash on the SOURCE path is important to avoid creating a nested directory

echo "Starting sync FROM HPC..."

# Fix potential host key verification errors
ssh-keygen -R greene.hpc.nyu.edu > /dev/null 2>&1

rsync -avzP \
    --exclude='.git/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='notebooks/data/' \
    kl4747@greene.hpc.nyu.edu:/scratch/kl4747/Fourier-Neural-Operator-main/ ./

echo "Sync complete!"
