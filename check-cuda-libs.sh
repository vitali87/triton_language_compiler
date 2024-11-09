#!/bin/bash

echo "Checking CUDA symbolic links..."
readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1
readlink -f /usr/local/cuda

echo -e "\nChecking NVIDIA libraries..."
ldconfig -p | grep -i nvidia | grep -i cuda

echo -e "\nChecking actual CUDA installation..."
ls -la /usr/local/cuda/lib64/libcudart.so*
ls -la /usr/lib/cuda/lib64/libcudart.so*

echo -e "\nChecking NVIDIA driver version..."
cat /proc/driver/nvidia/version
