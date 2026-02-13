# Getting Started with Zenith

This guide will help you get started with the Zenith programming language.

## Prerequisites

Before building Zenith, you need:

1. **LLVM 21+ with MLIR**
   - Download from [LLVM releases](https://github.com/llvm/llvm-project/releases)
   - Or build from source

2. **C++20 Compiler**
   - GCC 10+ or Clang 12+ or MSVC 2019+

3. **CMake 3.20+**
   - Download from [cmake.org](https://cmake.org/download/)

4. **Ninja Build System** (recommended)
   - Install via package manager or from [ninja-build.org](https://ninja-build.org/)

## Building LLVM with MLIR (Optional)

If you don't have LLVM 21 with MLIR, build it from source:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-21.0.0  # or latest stable

mkdir build && cd build
cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_INSTALL_PREFIX=/path/to/install

ninja
ninja install
```

This will take 30-60 minutes depending on your system.

## Building Zenith

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/zenith.git
cd zenith
```

### 2. Set Environment Variables

```bash
export LLVM_DIR=/path/to/llvm/install/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/install/lib/cmake/mlir
```

Or add to your `~/.bashrc` or `~/.zshrc`.

### 3. Build with the Build Script

```bash
./build.sh
```

Or manually:

```bash
mkdir build && cd build
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR=$LLVM_DIR \
    -DMLIR_DIR=$MLIR_DIR

ninja
```

### 4. Run Tests

```bash
cd build
ninja check-zenith
```

## Your First Zenith Program

### Example 1: Basic Arithmetic

Create a file `hello.mlir`:

```mlir
// hello.mlir
func.func @main() {
    %c10 = zenith.constant 10 : i32
    %c32 = zenith.constant 32 : i32
    
    %sum = zenith.add %c10, %c32 : i32
    zenith.print %sum : i32
    
    return
}
```

Run it through the optimizer:

```bash
./build/bin/zenith-opt hello.mlir
```

### Example 2: Functions

Create `functions.mlir`:

```mlir
func.func @square(%x: i32) -> i32 {
    %result = zenith.mul %x, %x : i32
    return %result : i32
}

func.func @main() {
    %c5 = zenith.constant 5 : i32
    %result = zenith.call @square(%c5) : (i32) -> i32
    zenith.print %result : i32
    return
}
```

Run with optimization:

```bash
./build/bin/zenith-opt --zenith-constant-fold --zenith-inline functions.mlir
```

## Project Structure

```
zenith/
├── include/Zenith/       # Public headers
│   ├── Dialect/          # Dialect definitions
│   └── Passes/           # Pass definitions
├── lib/                  # Implementation
│   ├── Dialect/          # Dialect implementation
│   └── Passes/           # Pass implementation
├── tools/                # Command-line tools
│   └── zenith-opt/       # Optimizer driver
├── test/                 # Test suite
├── examples/             # Example programs
└── docs/                 # Documentation
```

## Using zenith-opt

`zenith-opt` is the main optimization tool for Zenith.

### Basic Usage

```bash
zenith-opt [options] input.mlir
```

### Common Options

- `--help`: Show all available options
- `-o output.mlir`: Write output to file
- `--print-ir-after-all`: Print IR after each pass
- `--mlir-timing`: Show pass timing information

### Optimization Passes

- `--zenith-constant-fold`: Fold constants
- `--zenith-inline`: Inline function calls
- `--zenith-arith-opt`: Optimize arithmetic
- `--zenith-lower-to-llvm`: Lower to LLVM dialect

### Example Pipeline

```bash
zenith-opt \
    --zenith-constant-fold \
    --zenith-inline \
    --zenith-arith-opt \
    --zenith-lower-to-llvm \
    input.mlir -o output.mlir
```

## IDE Setup

### Visual Studio Code

1. Install the MLIR extension
2. Add to `.vscode/settings.json`:

```json
{
    "mlir.server_path": "/path/to/llvm/build/bin/mlir-lsp-server",
    "files.associations": {
        "*.mlir": "mlir",
        "*.zen": "zenith"
    }
}
```

### CLion

1. Open the project
2. CLion will automatically detect CMake
3. Set CMake options in Settings > Build > CMake

### Vim/Neovim

Install vim-mlir plugin:

```vim
Plug 'Superty/vim-mlir'
```

## Next Steps

1. **Read the Documentation**
   - [Language Reference](docs/language-reference.md)
   - [Dialect Reference](docs/dialect.md)
   - [Pass Reference](docs/passes.md)

2. **Explore Examples**
   - Check out the `examples/` directory
   - Run and modify existing examples

3. **Contribute**
   - Read [CONTRIBUTING.md](CONTRIBUTING.md)
   - Pick an issue labeled "good first issue"
   - Submit a pull request

4. **Join the Community**
   - GitHub Discussions
   - Discord server
   - Mailing list

## Troubleshooting

### CMake can't find LLVM

**Problem:** `Could not find a package configuration file provided by "MLIR"`

**Solution:** Make sure LLVM_DIR and MLIR_DIR are set correctly:

```bash
export LLVM_DIR=/path/to/llvm/build/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/build/lib/cmake/mlir
```

### Compilation errors

**Problem:** C++ compilation errors

**Solution:** 
- Make sure you have C++20 support
- Update your compiler to GCC 10+ or Clang 12+

### Link errors

**Problem:** Undefined references during linking

**Solution:**
- Rebuild LLVM with `-DLLVM_ENABLE_PROJECTS="mlir"`
- Make sure all MLIR libraries are linked

## Resources

- **Official Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **MLIR Documentation**: https://mlir.llvm.org/
- **LLVM Documentation**: https://llvm.org/docs/

## Getting Help

If you encounter issues:

1. Check the [FAQ](docs/faq.md)
2. Search existing [GitHub Issues](https://github.com/yourusername/zenith/issues)
3. Ask on [GitHub Discussions](https://github.com/yourusername/zenith/discussions)
4. Join our Discord server

Happy coding with Zenith! 🚀

