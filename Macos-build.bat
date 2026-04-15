#!/bin/bash

echo "=========================================="
echo "   SuperWeb Cluster - macOS Compiler"
echo "=========================================="

# 1.  macOS CPU 
echo ""
echo "[1/2]  macOS CPU (clang++)..."
cd compute_node/performance_metrics/fixed_matrix_vector_multiplication/cpu/macos
mkdir -p build
clang++ ../fmvm_cpu_macos.cpp -std=c++20 -O3 -ffast-math -pthread -o build/fmvm_cpu_macos
cd - > /dev/null

# 2. macOS Metal (GPU) 
echo ""
echo "[2/2]  macOS Metal ..."
cd compute_node/performance_metrics/fixed_matrix_vector_multiplication/metal
mkdir -p build

# 2.1  Metal  (Metal -> AIR -> Metallib)
echo "  ->  Metal AIR ..."
xcrun --sdk macosx metal -c ../fmvm_metal_kernels.metal -o build/fmvm_metal_kernels.air
xcrun --sdk macosx metallib build/fmvm_metal_kernels.air -o build/fmvm_metal_kernels.metallib

# 2.2  Metal Main (Objective-C++)
echo "  ->  Metal Main..."
xcrun --sdk macosx clang++ ../fmvm_metal_runner.mm -std=c++20 -O3 -fobjc-arc \
    -framework Foundation -framework Metal \
    -Wl,-sectcreate,__DATA,__metallib,build/fmvm_metal_kernels.metallib \
    -o build/fmvm_metal_runner

cd - > /dev/null

echo ""
echo "=========================================="
echo "Finished！"