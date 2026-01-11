# Zenith Language - fn Keyword and Effect System

## Overview

This document describes the `fn` keyword and effect system additions to the Zenith programming language.

## fn Keyword

The `fn` keyword is used to declare pure functions that may not have side effects. It supports three syntaxes:

### 1. Traditional Syntax

```zenith
fn name(params) -> ret {
    // function body
}
```

Example:
```zenith
fn square(x: int) -> int {
    x * x
}
```

### 2. Type Signature Only

This syntax declares a function signature without providing an implementation, useful for interfaces or external function declarations:

```zenith
fn name : type -> type
```

Example:
```zenith
fn identity : int -> int
```

### 3. Lambda-Style Expression

Short-hand syntax for simple functions with a single expression:

```zenith
fn name (params) = expression
```

Example:
```zenith
fn cube(x: int) = x ** 3
```

## Effect/Monad System

Functions declared with `fn` can specify effects or monads they may produce using the `|` operator in type signatures:

```zenith
fn name : type -> type | Effect1, Effect2
```

Examples:
```zenith
// Function that performs IO
fn readFile : str -> str | IO

// Function that may throw exceptions and perform IO
fn processData : int -> int | IO, Exception
```

## Procedures vs Functions

### proc - Procedures with Side Effects

Procedures are declared with the `proc` keyword and may have side effects:

```zenith
proc writesomething() {
    print "HELLO!"
}
```

### fn - Pure Functions

Functions declared with `fn` may not have side effects (beyond declared effects). They are:
- Pure by default
- Cannot modify external state
- Parameters are immutable
- Return values depend only on input parameters (plus declared effects)

## Type System

### Unconstrained Types

Basic types like `int` are unconstrained by default, meaning they accept any value of that type:

```zenith
x: int = 42        // Any integer
y: int = -100      // Including negative
z: int = 999999    // Including large values
```

### Constrained Types

Types can have constraints applied using dependent types or ranges:

```zenith
x: int[0..100] = 50           // Range constraint
y: int @ positive = 42         // Named constraint
```

## Operators

### Exponentiation

The `**` operator is used for exponentiation:

```zenith
fn square(x: int) = x ** 2
result = 2 ** 8  // 256
```

## Implementation Details

### Grammar Changes

1. **Lexer (ZenithLexer.g4)**:
   - Added `FN` keyword
   - Added `POWER` token for `**` operator

2. **Parser (ZenithParser.g4)**:
   - Added `fn_declaration` rule supporting three syntaxes
   - Added `function_type_signature` for type-only declarations
   - Added `effect_list` for effect/monad annotations
   - Added `power_expr` for exponentiation operator

### HIR Changes

1. **HirFunc Structure**:
   - Added `Fn` to `Kind` enum
   - Added `effects` vector for effect annotations
   - Added `isTypeSignatureOnly` flag
   - Added `paramType` for type signature declarations

2. **HirExpr Structure**:
   - Added `Power` to `BinOpKind` enum

## Examples

See `examples_fn.zen` for complete examples demonstrating all features.
