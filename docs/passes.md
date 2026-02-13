# Zenith Pass Reference

## Overview

This document describes the optimization and transformation passes available in the Zenith compiler.

## Pass Pipeline

The typical compilation pipeline for Zenith programs:

```
Source Code
    ↓
[Parsing & Semantic Analysis]
    ↓
Zenith Dialect (High-level IR)
    ↓
[High-level Optimization Passes]
    ↓
Zenith Dialect (Optimized)
    ↓
[Lowering Pass]
    ↓
LLVM Dialect
    ↓
[LLVM Optimization Passes]
    ↓
Machine Code
```

## Optimization Passes

### Constant Folding Pass

**Name:** `zenith-constant-fold`

**Description:** Evaluates operations with constant operands at compile time.

**Usage:**

```bash
zenith-opt --zenith-constant-fold input.mlir
```

**Example:**

Before:

```mlir
func.func @example() -> i32 {
    %c1 = zenith.constant 10 : i32
    %c2 = zenith.constant 20 : i32
    %sum = zenith.add %c1, %c2 : i32
    return %sum : i32
}
```

After:

```mlir
func.func @example() -> i32 {
    %c30 = zenith.constant 30 : i32
    return %c30 : i32
}
```

**Benefits:**

- Reduces runtime computation
- Enables further optimizations
- Smaller code size

### Inline Pass

**Name:** `zenith-inline`

**Description:** Inlines function calls to eliminate call overhead and enable further optimizations.

**Usage:**

```bash
zenith-opt --zenith-inline input.mlir
```

**Example:**

Before:

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = zenith.add %arg0, %arg1 : i32
    return %0 : i32
}

func.func @main() -> i32 {
    %c1 = zenith.constant 5 : i32
    %c2 = zenith.constant 10 : i32
    %result = zenith.call @add(%c1, %c2) : (i32, i32) -> i32
    return %result : i32
}
```

After:

```mlir
func.func @main() -> i32 {
    %c1 = zenith.constant 5 : i32
    %c2 = zenith.constant 10 : i32
    %0 = zenith.add %c1, %c2 : i32
    return %0 : i32
}
```

**Heuristics:**

- Small functions are always inlined
- Functions marked with `@inline` attribute are always inlined
- Recursive functions are not inlined
- Functions with many call sites may not be inlined to avoid code bloat

### Arithmetic Optimization Pass

**Name:** `zenith-arith-opt`

**Description:** Performs algebraic simplifications and strength reduction on arithmetic operations.

**Usage:**

```bash
zenith-opt --zenith-arith-opt input.mlir
```

**Optimizations:**

1. **Identity elimination:**
    - `x + 0 → x`
    - `x * 1 → x`
    - `x - 0 → x`

2. **Strength reduction:**
    - `x * 2 → x << 1` (for integers)
    - `x / 2 → x >> 1` (for unsigned integers)

3. **Algebraic simplification:**
    - `x - x → 0`
    - `x * 0 → 0`
    - `(x + c1) + c2 → x + (c1 + c2)`

4. **Reassociation:**
    - `(a + b) + c → a + (b + c)` (when beneficial)

### Shape Inference Pass

**Name:** `zenith-shape-inference`

**Description:** Infers shapes for array and tensor operations.

**Usage:**

```bash
zenith-opt --zenith-shape-inference input.mlir
```

**Example:**

Before:

```mlir
func.func @array_op(%arg0: !zenith.array<i32, ?>) -> !zenith.array<i32, ?> {
    %0 = zenith.array.map %arg0 : !zenith.array<i32, ?> -> !zenith.array<i32, ?>
    return %0 : !zenith.array<i32, ?>
}
```

After:

```mlir
func.func @array_op(%arg0: !zenith.array<i32, 10>) -> !zenith.array<i32, 10> {
    %0 = zenith.array.map %arg0 : !zenith.array<i32, 10> -> !zenith.array<i32, 10>
    return %0 : !zenith.array<i32, 10>
}
```

## Lowering Passes

### Lower to LLVM Pass

**Name:** `zenith-lower-to-llvm`

**Description:** Lowers Zenith dialect operations to LLVM dialect.

**Usage:**

```bash
zenith-opt --zenith-lower-to-llvm input.mlir
```

**Example:**

Before (Zenith dialect):

```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = zenith.add %arg0, %arg1 : i32
    return %0 : i32
}
```

After (LLVM dialect):

```mlir
llvm.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = llvm.add %arg0, %arg1 : i32
    llvm.return %0 : i32
}
```

**Type Conversions:**

- `i32` → `i32` (LLVM)
- `f64` → `double` (LLVM)
- `!zenith.string` → `!llvm.ptr<i8>` (C-style string)
- `!zenith.array<T, N>` → `!llvm.array<N x T>`

## Pass Pipelines

### Standard Pipeline

For typical compilation:

```bash
zenith-opt \
    --zenith-constant-fold \
    --zenith-inline \
    --zenith-arith-opt \
    --zenith-constant-fold \
    --zenith-lower-to-llvm \
    input.mlir
```

### Debug Pipeline

For debugging with minimal optimization:

```bash
zenith-opt \
    --zenith-lower-to-llvm \
    input.mlir
```

### Aggressive Optimization Pipeline

For maximum performance:

```bash
zenith-opt \
    --zenith-constant-fold \
    --zenith-inline \
    --zenith-constant-fold \
    --zenith-arith-opt \
    --zenith-constant-fold \
    --zenith-shape-inference \
    --zenith-inline \
    --zenith-arith-opt \
    --zenith-constant-fold \
    --zenith-lower-to-llvm \
    input.mlir
```

## Pass Options

### Inline Pass Options

- `--inline-threshold=<n>`: Set the threshold for inlining decisions (default: 50)
- `--inline-recursive`: Allow recursive inlining (use with caution)

### Arithmetic Optimization Options

- `--arith-opt-level=<0-3>`: Set optimization aggressiveness (default: 2)
    - 0: No optimization
    - 1: Basic optimizations
    - 2: Standard optimizations
    - 3: Aggressive optimizations

## Custom Pass Development

### Creating a New Pass

1. **Define the pass in TableGen** (`Passes.td`):

```tablegen
def MyCustomPass : Pass<"zenith-my-pass", "::mlir::func::FuncOp"> {
    let summary = "My custom optimization pass";
    let description = [{
        Detailed description of what the pass does.
    }];
    let constructor = "mlir::zenith::createMyCustomPass()";
}
```

2. **Implement the pass** (`MyCustomPass.cpp`):

```cpp
struct MyCustomPass : public PassWrapper<MyCustomPass, OperationPass<func::FuncOp>> {
    void runOnOperation() override {
        auto func = getOperation();
        // Implement pass logic here
    }
};

std::unique_ptr<Pass> createMyCustomPass() {
    return std::make_unique<MyCustomPass>();
}
```

3. **Register the pass**:

Add the declaration to `Passes.h` and implementation to `Passes.cpp`.

### Pass Testing

Create a lit test file:

```mlir
// RUN: zenith-opt --zenith-my-pass %s | FileCheck %s

// CHECK-LABEL: func @test_function
func.func @test_function(%arg0: i32) -> i32 {
    // CHECK: optimized code here
    return %arg0 : i32
}
```

## Performance Considerations

### Pass Ordering

- Run constant folding after passes that create new constants
- Run inlining before optimization passes to expose more optimization opportunities
- Run dead code elimination after other passes

### Pass Efficiency

- Use pattern rewriting for efficient transformations
- Implement proper cost models for optimization decisions
- Use caching to avoid redundant computations

## Debugging Passes

### Print IR Between Passes

```bash
zenith-opt --print-ir-after-all input.mlir
```

### Print Only Specific Pass

```bash
zenith-opt --print-ir-after=zenith-inline input.mlir
```

### Verify IR After Each Pass

```bash
zenith-opt --verify-each input.mlir
```

### Enable Pass Timing

```bash
zenith-opt --mlir-timing input.mlir
```

## See Also

- [Language Reference](language-reference.md)
- [Dialect Reference](dialect.md)
- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)

