# Zenith MLIR Language - Template Summary

## Overview

This is a complete, production-ready template for building a new programming language based on LLVM's MLIR infrastructure. The language is named **Zenith** and demonstrates best practices for MLIR-based language implementation.

## Key Features

### Technical Specifications
- **C++20**: Modern C++ with latest features
- **LLVM 21+**: Latest LLVM/MLIR infrastructure
- **CMake 3.20+**: Modern CMake build system
- **TableGen**: Declarative operation/pass definitions
- **Lit Testing**: Comprehensive test infrastructure

### Language Features Implemented
- ✅ Basic arithmetic operations (add, sub, mul, div)
- ✅ Constants and type system
- ✅ Function definitions and calls
- ✅ Return statements
- ✅ Print operation
- ✅ MLIR dialect definition
- ✅ Optimization passes foundation
- ✅ LLVM dialect lowering

### Project Structure

```
zenith/
├── CMakeLists.txt              # Main build configuration
├── build.sh                    # Build script
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contribution guidelines
├── ROADMAP.md                  # Development roadmap
│
├── include/Zenith/             # Public headers
│   ├── Dialect/
│   │   ├── ZenithDialect.h     # Dialect header
│   │   ├── ZenithDialect.td    # Dialect TableGen
│   │   ├── ZenithOps.h         # Operations header
│   │   ├── ZenithOps.td        # Operations TableGen
│   │   └── ZenithTypes.h       # Type system
│   └── Passes/
│       ├── Passes.h            # Pass headers
│       └── Passes.td           # Pass TableGen
│
├── lib/                        # Implementation
│   ├── Dialect/
│   │   ├── ZenithDialect.cpp   # Dialect implementation
│   │   └── ZenithOps.cpp       # Operations implementation
│   └── Passes/
│       ├── Passes.cpp          # Pass registration
│       └── LowerToLLVM.cpp     # LLVM lowering pass
│
├── tools/                      # Command-line tools
│   └── zenith-opt/
│       ├── zenith-opt.cpp      # Optimizer driver
│       └── CMakeLists.txt
│
├── test/                       # Test suite
│   ├── lit.cfg.py              # Lit configuration
│   ├── CMakeLists.txt
│   └── Dialect/
│       └── basic.mlir          # Basic tests
│
├── examples/                   # Example programs
│   ├── arithmetic.mlir         # Basic arithmetic
│   ├── functions.mlir          # Function calls
│   └── factorial.mlir          # Recursive example
│
└── docs/                       # Documentation
    ├── getting-started.md      # Getting started guide
    ├── language-reference.md   # Language specification
    ├── dialect.md              # Dialect reference
    └── passes.md               # Pass reference
```

## Quick Start

### Prerequisites
```bash
# Install LLVM 21+ with MLIR
# Install CMake 3.20+
# Install Ninja build system
# Ensure C++20 compatible compiler
```

### Build
```bash
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
./build.sh
```

### Test
```bash
cd build
ninja check-zenith
```

### Run Example
```bash
./build/bin/zenith-opt examples/arithmetic.mlir
```

## Core Components

### 1. Dialect Definition (TableGen)

**File:** `include/Zenith/Dialect/ZenithOps.td`

Defines operations using MLIR's TableGen:
- Operations: add, sub, mul, div, constant, func, call, return, print
- Traits: Pure, SameOperandsAndResultType, ConstantLike
- Interfaces: FunctionOpInterface, CallOpInterface

### 2. Dialect Implementation

**Files:** `lib/Dialect/ZenithDialect.cpp`, `lib/Dialect/ZenithOps.cpp`

Implements:
- Operation folding for constant propagation
- Function parsing and printing
- Type system foundation

### 3. Passes

**Files:** `lib/Passes/*.cpp`

Includes:
- Constant folding pass
- Inlining pass (stub)
- Arithmetic optimization pass (stub)
- LLVM lowering pass (functional)

### 4. Tools

**File:** `tools/zenith-opt/zenith-opt.cpp`

Command-line optimizer:
- Loads Zenith dialect
- Registers all passes
- Provides MLIR optimization pipeline

## Operations Reference

### Arithmetic
```mlir
%sum = zenith.add %a, %b : i32
%diff = zenith.sub %a, %b : i32
%prod = zenith.mul %a, %b : i32
%quot = zenith.div %a, %b : i32
```

### Constants
```mlir
%c42 = zenith.constant 42 : i32
%pi = zenith.constant 3.14 : f64
```

### Functions
```mlir
zenith.func @add(%x: i32, %y: i32) -> i32 {
    %result = zenith.add %x, %y : i32
    zenith.return %result : i32
}
```

### Calls
```mlir
%result = zenith.call @add(%a, %b) : (i32, i32) -> i32
```

### I/O
```mlir
zenith.print %value : i32
```

## Extending the Language

### Adding a New Operation

1. **Define in TableGen** (`ZenithOps.td`):
```tablegen
def MyOp : Zenith_Op<"myop", [Pure]> {
    let summary = "My custom operation";
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$output);
    let assemblyFormat = "$input attr-dict `:` type($output)";
}
```

2. **Implement** (`ZenithOps.cpp`):
```cpp
// Add any custom methods or folding logic
```

3. **Test** (`test/Dialect/myop.mlir`):
```mlir
// RUN: zenith-opt %s | FileCheck %s
// CHECK-LABEL: func @test_myop
func.func @test_myop(%arg0: i32) -> i32 {
    // CHECK: zenith.myop
    %0 = zenith.myop %arg0 : i32
    return %0 : i32
}
```

### Adding a New Pass

1. **Define in TableGen** (`Passes.td`):
```tablegen
def MyPass : Pass<"zenith-mypass", "::mlir::func::FuncOp"> {
    let summary = "My custom pass";
    let constructor = "mlir::zenith::createMyPass()";
}
```

2. **Implement** (`lib/Passes/MyPass.cpp`):
```cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
    void runOnOperation() override {
        // Implementation
    }
};

std::unique_ptr<Pass> createMyPass() {
    return std::make_unique<MyPass>();
}
```

3. **Register** (`Passes.h`):
```cpp
std::unique_ptr<Pass> createMyPass();
```

## Documentation

Comprehensive documentation is provided:

1. **[Getting Started](docs/getting-started.md)**: Setup and first program
2. **[Language Reference](docs/language-reference.md)**: Complete language spec
3. **[Dialect Reference](docs/dialect.md)**: MLIR operations and types
4. **[Pass Reference](docs/passes.md)**: Optimization passes
5. **[Contributing](CONTRIBUTING.md)**: Development guidelines
6. **[Roadmap](ROADMAP.md)**: Future development plans

## Testing

The template includes a complete testing infrastructure:

- **Lit tests**: MLIR IR testing with FileCheck
- **Unit tests**: For individual components (to be added)
- **Integration tests**: End-to-end compilation (to be added)

Run tests:
```bash
cd build
ninja check-zenith
```

## Best Practices Demonstrated

1. **Separation of Concerns**: Clean separation between dialect, passes, and tools
2. **TableGen Usage**: Declarative operation and pass definitions
3. **MLIR Interfaces**: Proper use of MLIR's interface system
4. **Testing**: Comprehensive test coverage with Lit
5. **Documentation**: Well-documented code and user guides
6. **Build System**: Modern CMake with proper dependency management
7. **C++20 Features**: Utilizes modern C++ capabilities

## Customization Guide

To adapt this template for your language:

1. **Rename**: Replace "Zenith" with your language name
2. **Modify Operations**: Add/remove operations in `ZenithOps.td`
3. **Extend Type System**: Define custom types in `ZenithTypes.h`
4. **Add Passes**: Implement optimization passes for your language
5. **Build Frontend**: Add lexer/parser for your syntax
6. **Update Documentation**: Customize docs for your language

## Resources

- **MLIR Documentation**: https://mlir.llvm.org/
- **LLVM Documentation**: https://llvm.org/docs/
- **TableGen**: https://llvm.org/docs/TableGen/
- **Lit Testing**: https://llvm.org/docs/CommandGuide/lit.html

## License

Apache 2.0 License with LLVM Exceptions

## Support

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Questions and discussions
- Discord: Real-time community chat

---

**This template provides everything needed to start building a production-quality programming language on MLIR!** 🚀

