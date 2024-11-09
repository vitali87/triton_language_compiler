#!/bin/bash

echo "Looking for CUDA installation..."

echo -e "\nChecking common CUDA paths:"
for path in /usr/local/cuda* /usr/lib/cuda* /opt/cuda*; do
    if [ -d "$path" ]; then
        echo "Found CUDA directory: $path"
        if [ -d "$path/bin" ]; then
            echo "  - Has bin directory"
            if [ -f "$path/bin/nvcc" ]; then
                echo "  - Found nvcc"
            fi
        fi
        if [ -d "$path/lib64" ]; then
            echo "  - Has lib64 directory"
            if [ -f "$path/lib64/libcudart.so" ]; then
                echo "  - Found libcudart.so"
            fi
        fi
    fi
done

echo -e "\nChecking for CUDA libraries:"
for lib in $(ldconfig -p | grep -i cuda); do
    echo $lib
done

echo -e "\nChecking nvidia-smi version:"
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader

echo -e "\nChecking nvcc version:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found in PATH"
fi
