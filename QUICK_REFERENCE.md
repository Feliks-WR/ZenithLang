# Zenith Quick Reference

## Building

```bash
# Set environment
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir

# Build
./build.sh

# Run tests
cd build && ninja check-zenith
```

## Operations

| Operation | Syntax                                  | Example                                             |
|-----------|-----------------------------------------|-----------------------------------------------------|
| Constant  | `zenith.constant <value> : <type>`      | `%c = zenith.constant 42 : i32`                     |
| Add       | `zenith.add %a, %b : <type>`            | `%sum = zenith.add %x, %y : i32`                    |
| Subtract  | `zenith.sub %a, %b : <type>`            | `%diff = zenith.sub %x, %y : i32`                   |
| Multiply  | `zenith.mul %a, %b : <type>`            | `%prod = zenith.mul %x, %y : i32`                   |
| Divide    | `zenith.div %a, %b : <type>`            | `%quot = zenith.div %x, %y : i32`                   |
| Function  | `zenith.func @name(...) -> ... { ... }` | See below                                           |
| Call      | `zenith.call @func(...) : (...) -> ...` | `%r = zenith.call @add(%a, %b) : (i32, i32) -> i32` |
| Return    | `zenith.return [%value : <type>]`       | `zenith.return %result : i32`                       |
| Print     | `zenith.print %value : <type>`          | `zenith.print %x : i32`                             |

## Function Example

```mlir
zenith.func @square(%x: i32) -> i32 {
    %result = zenith.mul %x, %x : i32
    zenith.return %result : i32
}

func.func @main() {
    %c5 = zenith.constant 5 : i32
    %result = zenith.call @square(%c5) : (i32) -> i32
    zenith.print %result : i32
    return
}
```

## Passes

| Pass          | Flag                     | Purpose                            |
|---------------|--------------------------|------------------------------------|
| Constant Fold | `--zenith-constant-fold` | Evaluate constants at compile-time |
| Inline        | `--zenith-inline`        | Inline function calls              |
| Arith Opt     | `--zenith-arith-opt`     | Optimize arithmetic operations     |
| Lower to LLVM | `--zenith-lower-to-llvm` | Convert to LLVM dialect            |

## Pass Pipeline Example

```bash
zenith-opt \
    --zenith-constant-fold \
    --zenith-inline \
    --zenith-arith-opt \
    --zenith-lower-to-llvm \
    input.mlir
```

## Common zenith-opt Options

| Option                 | Description               |
|------------------------|---------------------------|
| `-o <file>`            | Output to file            |
| `--print-ir-after-all` | Print IR after each pass  |
| `--mlir-timing`        | Show timing information   |
| `--verify-each`        | Verify IR after each pass |
| `--help`               | Show all options          |

## Types

| Type                                    | Description     | Example        |
|-----------------------------------------|-----------------|----------------|
| `i1`, `i8`, `i16`, `i32`, `i64`, `i128` | Signed integers | `42 : i32`     |
| `f16`, `f32`, `f64`                     | Floating point  | `3.14 : f64`   |
| `index`                                 | Index type      | `%idx : index` |

## Project Structure

```
zenith/
├── include/Zenith/      # Headers (.h, .td)
├── lib/                 # Implementation (.cpp)
├── tools/zenith-opt/    # Optimizer tool
├── test/                # Tests (.mlir)
├── examples/            # Examples (.mlir)
└── docs/                # Documentation (.md)
```

## File Extensions

| Extension | Type                             |
|-----------|----------------------------------|
| `.mlir`   | MLIR intermediate representation |
| `.zen`    | Zenith source code (future)      |
| `.td`     | TableGen definition              |

## Debugging

```bash
# Print IR after specific pass
zenith-opt --print-ir-after=zenith-inline input.mlir

# Enable all diagnostics
zenith-opt --mlir-print-debuginfo input.mlir

# Timing information
zenith-opt --mlir-timing --mlir-timing-display=tree input.mlir
```

## Adding New Operations

1. Define in `include/Zenith/Dialect/ZenithOps.td`
2. Implement in `lib/Dialect/ZenithOps.cpp` (if needed)
3. Add tests in `test/Dialect/`
4. Update documentation

## Adding New Passes

1. Define in `include/Zenith/Passes/Passes.td`
2. Implement in `lib/Passes/`
3. Declare in `include/Zenith/Passes/Passes.h`
4. Add tests in `test/Transforms/`
5. Update documentation

## Resources

- Docs: `docs/`
- Examples: `examples/`
- MLIR: https://mlir.llvm.org/
- GitHub: https://github.com/yourusername/zenith

## Common Issues

**Can't find LLVM:**

```bash
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir
```

**C++20 errors:**

- Update compiler to GCC 10+, Clang 12+, or MSVC 2019+

**Link errors:**

- Rebuild LLVM with `-DLLVM_ENABLE_PROJECTS="mlir"`

## Getting Help

- Documentation: `docs/`
- GitHub Issues
- GitHub Discussions
- Discord community

