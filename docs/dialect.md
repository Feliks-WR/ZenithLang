# Zenith Dialect Reference

## Overview

The Zenith dialect is the high-level dialect for the Zenith programming language. It provides operations that closely match the language semantics before being lowered to LLVM IR.

## Operations

### Constant Operations

#### `zenith.constant`

Creates a constant value.

**Syntax:**
```mlir
%result = zenith.constant <value> : <type>
```

**Examples:**
```mlir
%0 = zenith.constant 42 : i32
%1 = zenith.constant 3.14 : f64
%2 = zenith.constant true : i1
```

**Properties:**
- Pure: This operation has no side effects
- ConstantLike: Can be folded at compile time

### Arithmetic Operations

#### `zenith.add`

Performs addition on two operands.

**Syntax:**
```mlir
%result = zenith.add %lhs, %rhs : <type>
```

**Examples:**
```mlir
%sum = zenith.add %a, %b : i32
%fsum = zenith.add %x, %y : f64
```

**Properties:**
- Pure: No side effects
- Commutative: Order of operands doesn't matter
- Foldable: Can be constant folded

#### `zenith.sub`

Performs subtraction.

**Syntax:**
```mlir
%result = zenith.sub %lhs, %rhs : <type>
```

#### `zenith.mul`

Performs multiplication.

**Syntax:**
```mlir
%result = zenith.mul %lhs, %rhs : <type>
```

**Properties:**
- Pure: No side effects
- Commutative: Order of operands doesn't matter
- Foldable: Can be constant folded

#### `zenith.div`

Performs division.

**Syntax:**
```mlir
%result = zenith.div %lhs, %rhs : <type>
```

**Note:** Division by zero is undefined behavior.

### Control Flow Operations

#### `zenith.return`

Returns from a function.

**Syntax:**
```mlir
zenith.return                    // Return void
zenith.return %value : <type>    // Return value
```

**Examples:**
```mlir
func.func @void_function() {
    zenith.return
}

func.func @returns_int() -> i32 {
    %0 = zenith.constant 42 : i32
    zenith.return %0 : i32
}
```

### Function Operations

#### `zenith.func`

Defines a function.

**Syntax:**
```mlir
zenith.func @name(%arg0: <type>, ...) -> <return_type> {
    // function body
}
```

**Examples:**
```mlir
zenith.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = zenith.add %arg0, %arg1 : i32
    zenith.return %0 : i32
}

zenith.func @void_function() {
    zenith.return
}
```

**Attributes:**
- `sym_name`: Symbol name of the function
- `function_type`: Type signature of the function
- `arg_attrs`: Optional attributes for arguments
- `res_attrs`: Optional attributes for results

#### `zenith.call`

Calls a function.

**Syntax:**
```mlir
%results = zenith.call @function_name(%args) : (<arg_types>) -> <result_types>
```

**Examples:**
```mlir
%result = zenith.call @add(%a, %b) : (i32, i32) -> i32

zenith.call @print_message(%msg) : (!zenith.string) -> ()
```

### I/O Operations

#### `zenith.print`

Prints a value to stdout.

**Syntax:**
```mlir
zenith.print %value : <type>
```

**Examples:**
```mlir
%0 = zenith.constant 42 : i32
zenith.print %0 : i32

%1 = zenith.constant 3.14 : f64
zenith.print %1 : f64
```

## Types

### Built-in Types

The Zenith dialect uses MLIR's built-in types for most operations:

- **Integer types**: `i1`, `i8`, `i16`, `i32`, `i64`, `i128`
- **Floating-point types**: `f16`, `f32`, `f64`
- **Index type**: `index` (machine-word-sized integer)

### Custom Types

#### `zenith.string`

Represents a string value.

**Syntax:**
```mlir
!zenith.string
```

#### `zenith.array`

Represents an array type.

**Syntax:**
```mlir
!zenith.array<<element_type>, <size>>
```

**Example:**
```mlir
!zenith.array<i32, 10>  // Array of 10 i32 values
```

## Attributes

### Function Attributes

- `inline`: Suggests the function should be inlined
- `noinline`: Prevents function inlining
- `pure`: Indicates the function has no side effects
- `const`: Indicates the function is a compile-time constant

### Operation Attributes

- `overflow_flags`: Controls overflow behavior for arithmetic operations
  - `nsw`: No signed wrap
  - `nuw`: No unsigned wrap

## Lowering

The Zenith dialect is lowered to LLVM dialect through several passes:

1. **High-level optimizations**: Constant folding, inlining, etc.
2. **Type lowering**: Convert Zenith types to LLVM types
3. **Operation lowering**: Convert Zenith operations to LLVM operations
4. **Cleanup**: Remove dead code, simplify control flow

### Example Lowering

**Zenith Dialect:**
```mlir
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = zenith.add %arg0, %arg1 : i32
    return %0 : i32
}
```

**After Lowering to LLVM:**
```mlir
llvm.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = llvm.add %arg0, %arg1 : i32
    llvm.return %0 : i32
}
```

## Optimization Passes

### Available Passes

- **zenith-constant-fold**: Folds constant operations
- **zenith-inline**: Inlines function calls
- **zenith-arith-opt**: Optimizes arithmetic operations
- **zenith-shape-inference**: Infers shapes for array operations
- **zenith-lower-to-llvm**: Lowers to LLVM dialect

### Usage

```bash
zenith-opt --zenith-constant-fold --zenith-inline input.mlir
```

## Interface Implementations

The Zenith dialect operations implement several MLIR interfaces:

- **CallOpInterface**: For call operations
- **FunctionOpInterface**: For function definitions
- **InferTypeOpInterface**: For type inference
- **SideEffectInterface**: For tracking side effects
- **CastOpInterface**: For type casting operations

## Extension Points

The dialect is designed to be extensible:

1. **Custom types**: Add new types by registering them in `registerTypes()`
2. **Custom operations**: Define new operations in TableGen
3. **Custom attributes**: Register new attributes in `registerAttributes()`
4. **Custom passes**: Implement new optimization passes

## Best Practices

1. **Use high-level operations**: Keep operations at the highest abstraction level possible
2. **Leverage folding**: Implement fold methods for operations that can be constant-folded
3. **Use interfaces**: Implement MLIR interfaces to enable generic transformations
4. **Document operations**: Use TableGen's description fields to document operations
5. **Test thoroughly**: Add lit tests for each operation and pass

## See Also

- [Language Reference](language-reference.md)
- [Pass Reference](passes.md)
- [MLIR Documentation](https://mlir.llvm.org/)

