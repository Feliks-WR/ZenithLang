# Zenith Language Features

## Overview

Zenith is a modern, compiled programming language that transpiles to C for maximum performance and compatibility.

## Language Features

### 1. Functions

#### Basic Functions
```zenith
main() {
    return 0
}
```

#### Functions with Parameters
```zenith
add(a, b) {
    return a + b
}

multiply(x, y) {
    result = x * y
    return result
}
```

#### Function Calls
```zenith
main() {
    sum = add(5, 3)
    product = multiply(4, 7)
    return sum
}
```

### 2. Variables

Variables are automatically declared on first assignment with type inference:

```zenith
main() {
    x = 42          // int
    y = 3.14        // float
    name = "Alice"  // string
    return 0
}
```

### 3. Control Flow

#### If Statements
```zenith
main() {
    x = 10
    
    if x > 5 {
        return 1
    }
    
    return 0
}
```

#### If-Else
```zenith
check(n) {
    if n < 0 {
        return -1
    } else {
        return 1
    }
}
```

#### Nested Conditions
```zenith
classify(x) {
    if x < 0 {
        return -1
    } else {
        if x == 0 {
            return 0
        } else {
            return 1
        }
    }
}
```

#### While Loops
```zenith
countdown(n) {
    i = n
    while i > 0 {
        i = i - 1
    }
    return i
}
```

### 4. Print Statement

The `print` statement outputs values to stdout:

```zenith
main() {
    x = 42
    print x           // Prints: 42
    
    print "Hello!"    // Prints: Hello!
    
    y = 3.14
    print y           // Prints: 3.140000
    
    return 0
}
```

### 5. Operators

#### Arithmetic
- `+` Addition
- `-` Subtraction
- `*` Multiplication
- `/` Division
- `%` Modulo
- `**` Power (not yet implemented in codegen)

#### Comparison
- `==` Equal
- `!=` Not equal
- `<` Less than
- `<=` Less than or equal
- `>` Greater than
- `>=` Greater than or equal

#### Logical
- `&&` AND
- `||` OR
- `!` NOT

#### Bitwise
- `&` AND
- `|` OR
- `^` XOR
- `~` NOT
- `<<` Left shift
- `>>` Right shift

### 6. Data Types

#### Supported Types
- `int` - Integer numbers
- `float` - Floating-point numbers
- `string` - Text literals

#### Type Inference
Types are automatically inferred from the value:
```zenith
x = 42        // int
y = 3.14      // float
s = "text"    // string
```

### 7. Comments

```zenith
// Single-line comment

/*
   Multi-line
   comment
*/

main() {
    // This is a comment
    return 0  // End-of-line comment
}
```

## Example Programs

### Fibonacci Sequence
```zenith
fibonacci(n) {
    a = 0
    b = 1
    i = 0
    
    while i < n {
        print a
        temp = a + b
        a = b
        b = temp
        i = i + 1
    }
    
    return a
}

main() {
    print "Fibonacci sequence (first 10):"
    result = fibonacci(10)
    return 0
}
```

### Prime Number Checker
```zenith
is_prime(n) {
    if n < 2 {
        return 0
    }
    
    i = 2
    while i * i <= n {
        if n % i == 0 {
            return 0
        }
        i = i + 1
    }
    
    return 1
}

main() {
    num = 17
    if is_prime(num) {
        print "Prime!"
    } else {
        print "Not prime"
    }
    return 0
}
```

### Factorial Calculator
```zenith
factorial(n) {
    if n <= 1 {
        return 1
    }
    
    result = 1
    i = 2
    
    while i <= n {
        result = result * i
        i = i + 1
    }
    
    return result
}

main() {
    n = 5
    result = factorial(n)
    print result  // Prints: 120
    return 0
}
```

## Compilation

### Basic Usage
```bash
zenith program.zenith
./program
```

### Output Options
```bash
# Specify output file
zenith program.zenith -o myprogram

# Generate C code only (don't compile)
zenith program.zenith --emit-c

# Generate C code without compiling
zenith program.zenith --no-compile
```

## Language Philosophy

1. **Simple Syntax** - Clean, readable code without unnecessary keywords
2. **Type Inference** - Automatic type detection reduces verbosity
3. **Compiled to C** - Maximum performance and compatibility
4. **Python-like** - Familiar syntax for quick adoption
5. **Static Analysis Ready** - Clear grammar enables powerful tooling

## Future Features (Planned)

- [ ] Arrays and indexing
- [ ] Structs/Records
- [ ] String manipulation
- [ ] File I/O
- [ ] Standard library
- [ ] Module system
- [ ] Type annotations (optional)
- [ ] Pattern matching
- [ ] Enums
- [ ] Error handling

## Limitations (Current)

- No arrays yet
- No structs/classes
- Limited string operations
- No file I/O
- Single-file programs only
- All numbers default to `int`

## Performance

Zenith compiles to optimized C code, which is then compiled with GCC. This provides:

- **Near-C performance** - Minimal overhead
- **Cross-platform** - Works anywhere C does
- **Binary output** - Fast startup, no interpreter
- **Optimizable** - C compiler optimizations apply

## Examples Location

See the `examples/` directory for complete working programs:

- `calculator.zenith` - Multi-function calculator
- `fibonacci.zenith` - Fibonacci sequence generator
- `primes.zenith` - Prime number finder
- `demo.zenith` - Language features showcase

## Testing

Run the test suite:
```bash
cd build
ctest --output-on-failure
```

See [TESTING.md](TESTING.md) for comprehensive testing documentation.
