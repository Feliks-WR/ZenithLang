#!/bin/bash
set -e

echo "Building Zenith project with CMake..."

# Use CLion's bundled CMake 4.1.2
CMAKE_BIN="/home/abdullah/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/cmake"

if [ ! -f "$CMAKE_BIN" ]; then
    echo "Error: CLion's bundled CMake not found at $CMAKE_BIN"
    echo "Please update the path or install CMake 4.1.2+"
    exit 1
fi

echo "Using CMake: $CMAKE_BIN"
$CMAKE_BIN --version

# Set LLVM and MLIR paths
export LLVM_DIR="/usr/lib/llvm-20/lib/cmake/llvm"
export MLIR_DIR="/usr/lib/llvm-20/lib/cmake/mlir"

echo "LLVM_DIR: $LLVM_DIR"
echo "MLIR_DIR: $MLIR_DIR"

# Ensure ANTLR 4.13.2 JAR exists locally to match upgraded runtime
ANTLR_VERSION=4.13.2
ANTLR_JAR_NAME=antlr-${ANTLR_VERSION}-complete.jar
TOOLS_DIR="$(pwd)/tools"
mkdir -p "$TOOLS_DIR"
if [ ! -f "$TOOLS_DIR/$ANTLR_JAR_NAME" ]; then
  echo "Downloading ANTLR ${ANTLR_VERSION} into $TOOLS_DIR..."
  curl -L -o "$TOOLS_DIR/$ANTLR_JAR_NAME" "https://www.antlr.org/download/${ANTLR_JAR_NAME}"
fi

# Check if antlr4 is available (runtime headers still needed)
if ! command -v antlr4 &> /dev/null && ! command -v java &> /dev/null; then
    echo "Error: Neither antlr4 nor java found. Please install ANTLR4 or Java."
    echo "To install on Ubuntu/Debian:"
    echo "  sudo apt-get install antlr4 libantlr4-runtime-dev"
    echo "Or install Java and download ANTLR JAR manually."
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p cmake-build-debug
cd cmake-build-debug

# Configure with CMake
echo "Configuring with CMake..."
$CMAKE_BIN ..

# Create output directories that CMake might not create
mkdir -p include/Zenith
mkdir -p antlr4gen

# Build the project
echo "Building zenith-cli..."
$CMAKE_BIN --build . --target zenith-cli -j$(nproc)

echo ""
echo "Build complete! The executable is at: cmake-build-debug/zenith-cli"
echo "You can run it with: ./cmake-build-debug/zenith-cli <input_file>"
