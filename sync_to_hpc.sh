#!/bin/bash

# Sync local files to HPC using rsync
# Excludes git history, virtual environments, cache, and large data files to speed up transfer
# Note: The trailing slash on ./ is important (it means "contents of current dir")
# Note: The trailing slash on the destination directory ensures we sync INTO that folder

echo "Starting sync to HPC..."

# Fix potential host key verification errors
ssh-keygen -R greene.hpc.nyu.edu > /dev/null 2>&1

rsync -avzP \
    --exclude='.git/' \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='*.mat' \
    --exclude='*.gif' \
    ./ kl4747@greene.hpc.nyu.edu:/scratch/kl4747/Fourier-Neural-Operator-main/

echo "Sync complete!"

