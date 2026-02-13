#!/bin/bash

# Build script for Zenith language
# Requires LLVM 21+ with MLIR to be installed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Zenith Build Script${NC}"
echo "================================"

# Check if LLVM_DIR is set
if [ -z "$LLVM_DIR" ]; then
    echo -e "${YELLOW}Warning: LLVM_DIR not set${NC}"
    echo "Please set LLVM_DIR to your LLVM build directory:"
    echo "  export LLVM_DIR=/path/to/llvm/build/lib/cmake/llvm"
    echo "  export MLIR_DIR=/path/to/llvm/build/lib/cmake/mlir"
    echo ""
    echo "Attempting to find LLVM automatically..."

    # Try to find LLVM
    for dir in /usr/local/lib/cmake/llvm /usr/lib/cmake/llvm /opt/llvm/lib/cmake/llvm; do
        if [ -d "$dir" ]; then
            export LLVM_DIR="$dir"
            export MLIR_DIR="${dir/llvm/mlir}"
            echo -e "${GREEN}Found LLVM at: $LLVM_DIR${NC}"
            break
        fi
    done

    if [ -z "$LLVM_DIR" ]; then
        echo -e "${RED}Error: Could not find LLVM installation${NC}"
        exit 1
    fi
fi

# Create build directory
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$LLVM_DIR" \
    -DMLIR_DIR="$MLIR_DIR" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo "Building Zenith..."
ninja

# Run tests if requested
if [ "$1" == "test" ]; then
    echo "Running tests..."
    ninja check-zenith
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "To use Zenith:"
echo "  ./build/bin/zenith-opt examples/arithmetic.mlir"

