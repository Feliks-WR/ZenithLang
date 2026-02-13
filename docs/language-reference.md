# Zenith Language Reference

## Overview

Zenith is a statically-typed, high-performance programming language built on MLIR infrastructure. This document describes the language syntax, semantics, and features.

## Basic Syntax

### Comments

```zenith
// Single-line comment

/*
 * Multi-line comment
 */
```

### Variables

Variables in Zenith are declared with explicit types:

```zenith
var x: i32 = 42;
var y: f64 = 3.14159;
var name: string = "Zenith";
```

### Types

Zenith supports the following built-in types:

- **Integer types**: `i8`, `i16`, `i32`, `i64`, `i128`
- **Unsigned integer types**: `u8`, `u16`, `u32`, `u64`, `u128`
- **Floating-point types**: `f16`, `f32`, `f64`
- **Boolean**: `bool`
- **String**: `string`
- **Void**: `void` (for functions that don't return a value)

### Functions

Functions are declared with the `func` keyword:

```zenith
func add(x: i32, y: i32) -> i32 {
    return x + y;
}

func greet(name: string) {
    print("Hello, " + name + "!");
}

func main() {
    var result: i32 = add(5, 3);
    greet("World");
}
```

### Operators

#### Arithmetic Operators

- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Modulo: `%`

#### Comparison Operators

- Equal: `==`
- Not equal: `!=`
- Less than: `<`
- Greater than: `>`
- Less than or equal: `<=`
- Greater than or equal: `>=`

#### Logical Operators

- AND: `&&`
- OR: `||`
- NOT: `!`

### Control Flow

#### If-Else Statements

```zenith
func abs(x: i32) -> i32 {
    if (x < 0) {
        return -x;
    } else {
        return x;
    }
}
```

#### While Loops

```zenith
func countdown(n: i32) {
    while (n > 0) {
        print(n);
        n = n - 1;
    }
}
```

#### For Loops

```zenith
func sum_range(start: i32, end: i32) -> i32 {
    var total: i32 = 0;
    for (var i: i32 = start; i < end; i = i + 1) {
        total = total + i;
    }
    return total;
}
```

### Arrays

```zenith
var numbers: [i32; 5] = [1, 2, 3, 4, 5];
var first: i32 = numbers[0];
numbers[1] = 10;
```

### Structs

```zenith
struct Point {
    x: f64,
    y: f64
}

func distance(p1: Point, p2: Point) -> f64 {
    var dx: f64 = p2.x - p1.x;
    var dy: f64 = p2.y - p1.y;
    return sqrt(dx * dx + dy * dy);
}
```

### Enums

```zenith
enum Color {
    Red,
    Green,
    Blue,
    RGB(u8, u8, u8)
}

func is_red(c: Color) -> bool {
    match (c) {
        Color.Red => return true,
        _ => return false
    }
}
```

## Advanced Features

### Generics

```zenith
func max<T>(a: T, b: T) -> T {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
```

### Traits

```zenith
trait Printable {
    func to_string(self) -> string;
}

impl Printable for Point {
    func to_string(self) -> string {
        return "Point(" + str(self.x) + ", " + str(self.y) + ")";
    }
}
```

### Pattern Matching

```zenith
func describe_number(n: i32) -> string {
    match (n) {
        0 => return "zero",
        1 => return "one",
        n if n < 0 => return "negative",
        _ => return "other"
    }
}
```

## Memory Management

Zenith uses automatic memory management with reference counting and compile-time ownership analysis.

### Ownership

```zenith
func transfer_ownership() {
    var x: string = "hello";
    var y: string = x;  // x is moved to y
    // x is no longer accessible
}
```

### Borrowing

```zenith
func borrow_value(s: &string) {
    print(s);  // s is borrowed, not owned
}

func main() {
    var message: string = "Hello";
    borrow_value(&message);
    print(message);  // message is still valid
}
```

## Standard Library

Zenith provides a comprehensive standard library including:

- **io**: Input/output operations
- **math**: Mathematical functions
- **string**: String manipulation
- **collections**: Arrays, vectors, maps, sets
- **time**: Time and date utilities
- **fs**: File system operations
- **net**: Network programming

## Compiler Directives

```zenith
@inline
func fast_function() {
    // This function will be inlined
}

@no_optimize
func debug_function() {
    // This function won't be optimized
}
```

## Examples

### Factorial

```zenith
func factorial(n: i32) -> i32 {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}
```

### Fibonacci

```zenith
func fibonacci(n: i32) -> i32 {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
```

### Quicksort

```zenith
func quicksort(arr: &mut [i32]) {
    if (arr.len() <= 1) {
        return;
    }
    
    var pivot: i32 = arr[arr.len() / 2];
    var left: [i32] = [];
    var right: [i32] = [];
    
    for (var i: i32 = 0; i < arr.len(); i = i + 1) {
        if (arr[i] < pivot) {
            left.push(arr[i]);
        } else if (arr[i] > pivot) {
            right.push(arr[i]);
        }
    }
    
    quicksort(&mut left);
    quicksort(&mut right);
    
    // Combine results
    arr.clear();
    arr.extend(left);
    arr.push(pivot);
    arr.extend(right);
}
```

