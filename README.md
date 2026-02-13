# Zenith Programming Language

Zenith is a modern programming language built on top of MLIR (Multi-Level Intermediate Representation) infrastructure,
designed for high-performance computing with elegant syntax and powerful abstractions.

## Features

- **MLIR-Based**: Built on top of LLVM's MLIR infrastructure for powerful optimizations
- **Static Typing**: Strong static type system with type inference
- **Modern Syntax**: Clean and expressive syntax
- **Extensible**: Easy to add new operations and transformations
- **Performance**: Multiple optimization passes for high performance

## Project Structure

```
zenith/
├── CMakeLists.txt          # Main CMake configuration
├── include/
│   └── Zenith/
│       ├── Dialect/        # Dialect definitions
│       ├── Passes/         # Transformation passes
│       └── Parser/         # Parser and lexer
├── lib/
│   ├── Dialect/            # Dialect implementation
│   ├── Passes/             # Pass implementations
│   └── Parser/             # Parser implementation
├── tools/
│   ├── zenith-opt/         # Optimization tool
│   └── zenith-translate/   # Translation tool
├── test/                   # Tests
└── examples/               # Example programs
```

## Building

### Prerequisites

- CMake 3.20 or higher
- LLVM 21+ with MLIR enabled
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)

### Build Instructions

```bash
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/build/lib/cmake/llvm
ninja
```

## Quick Start

```zenith
// hello.zenith
func main() {
    print("Hello, Zenith!");
}
```

Compile and run:

```bash
./build/bin/zenith-opt hello.zenith
```

## Language Examples

### Basic Function

```zenith
func add(x: i32, y: i32) -> i32 {
    return x + y;
}
```

### Control Flow

```zenith
func factorial(n: i32) -> i32 {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}
```

### Variables and Types

```zenith
func example() {
    var x: i32 = 42;
    var y: f64 = 3.14;
    var name: string = "Zenith";
}
```

## Documentation

See the [docs](docs/) directory for detailed documentation:

- [Language Reference](docs/language-reference.md)
- [Dialect Reference](docs/dialect.md)
- [Pass Reference](docs/passes.md)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License with LLVM Exceptions.

