#!/bin/zsh

# Clear existing CUDA-related environment variables
unset CUDA_HOME
unset CUDA_PATH
unset LD_LIBRARY_PATH
unset CUDA_VISIBLE_DEVICES

# Set primary CUDA paths
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME

# Set up library paths
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu"

# Add CUDA bins to PATH
export PATH="$CUDA_HOME/bin:$PATH"

# Enable GPU device
export CUDA_VISIBLE_DEVICES=0

# Print configuration
echo "CUDA Configuration:"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_PATH=$CUDA_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PATH includes CUDA: $PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Test CUDA availability
if command -v nvcc &> /dev/null; then
    echo -e "\nNVCC version:"
    nvcc --version
else
    echo "Warning: nvcc not found in PATH"
fi

if [ -f "/proc/driver/nvidia/version" ]; then
    echo -e "\nNVIDIA driver version:"
    cat /proc/driver/nvidia/version
else
    echo "Warning: NVIDIA driver version info not found"
fi
