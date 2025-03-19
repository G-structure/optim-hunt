#!/bin/bash
# Exit on any error
set -e

echo "SETUP_START"

# Update package list
echo "Updating package list..."
apt-get update -y

# Install git-lfs
echo "Installing git-lfs..."
apt-get install -y git-lfs

# Set up SSH for git operations
echo "Configuring SSH for git..."
export GIT_SSH_COMMAND="ssh -i ${PWD}/.ssh/hf_key -o StrictHostKeyChecking=no"

# Create permanent models directory
echo "Creating models directory..."
mkdir -p /opt/models

# Check if model already exists
if [ -d "/opt/models/Llama-3.1-8B-Instruct" ]; then
    echo "Model already exists, skipping download..."
else
    # Clone the repository using the configured SSH with Git LFS
    echo "Setting up Git LFS..."
    git lfs install

    echo "Cloning Llama model repository..."
    git clone --filter=blob:none git@hf.co:meta-llama/Llama-3.1-8B-Instruct /opt/models/Llama-3.1-8B-Instruct

    echo "Pulling LFS objects..."
    cd /opt/models/Llama-3.1-8B-Instruct
    git lfs pull
    cd -
fi

# Create completion file
touch ${PWD}/.setup_complete

echo "Setup completed successfully"
