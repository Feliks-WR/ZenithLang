# CustomLang - MLIR-based Language

An immutable-first programming language compiler built with MLIR (Multi-Level Intermediate Representation) and ANTLR.

## Key Features

- **Immutable by Default**: All variables are immutable - no reassignment allowed
- **Flexible Type System**: Users define custom types; no hardcoded types
- **Pure Expressions**: No side effects in expression evaluation
- **Separate Lexer/Parser**: Clean ANTLR grammar separation for maintainability

## Project Structure

```
.
├── CMakeLists.txt              # Build configuration
├── grammar/
│   ├── CustomLangLexer.g4      # ANTLR lexer (tokens)
│   └── CustomLangParser.g4     # ANTLR parser (syntax rules)
├── include/
│   ├── Dialect.h               # Custom MLIR dialect definition
│   ├── Ops.h                   # Custom MLIR operations
│   ├── Types.h                 # Custom types
│   └── ASTBuilder.h            # AST to MLIR conversion
├── src/
│   ├── main.cpp                # Compiler entry point
│   ├── Dialect.cpp             # Dialect implementation
│   ├── Ops.cpp                 # Operations implementation
│   ├── Types.cpp               # Types implementation
│   └── ASTBuilder.cpp          # AST builder implementation
└── build/                      # Build output directory
```

## Architecture

### Grammar (ANTLR)

- **Lexer** (`CustomLangLexer.g4`): Tokenizes input (keywords, operators, literals)
- **Parser** (`CustomLangParser.g4`): Defines syntax with proper operator precedence
- **Separation**: Clean separation of lexical and syntax analysis

### Type System

- No hardcoded types - users define their own types as identifiers
- Example: `let x: i32 = 42` where `i32` is just a user-defined type name
- Supports array types: `i32[10]`
- Supports pointer/reference types: `&i32`

### Immutability Constraints

- Variables declared with `let` are immutable
- No assignment operator in expression context (only in initialization)
- All expressions are pure - no side effects allowed
- Variables cannot be reassigned after initialization

### Compiler Pipeline

1. **Lexical Analysis**: ANTLR Lexer tokenizes input
2. **Parsing**: ANTLR Parser builds parse tree
3. **AST Building**: ASTBuilder visitor converts parse tree to MLIR
4. **Code Generation**: MLIR handles optimization and backend code generation

## Dependencies

- LLVM/MLIR (>= 15.0)
- ANTLR4 C++ Runtime
- CMake (>= 3.20)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./build/customlang <input-file>
```

### Example Input File

```customlang
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn main() {
    let x: i32 = 10;
    let y: i32 = 20;
    let result: i32 = add(x, y);
    return result;
}
```

**Note**: All variables are immutable. The following would be a compile error:

```customlang
let x: i32 = 10;
x = 20;  // ERROR: cannot reassign immutable variable
```

## Language Features

### Implemented

- [x] Function declarations
- [x] Variable declarations (immutable)
- [x] Arithmetic expressions
- [x] Control flow (if/else, loops)
- [x] Type annotations
- [x] Comments (line and block)

### Planned

- [ ] Structs and custom types
- [ ] Immutable data structures
- [ ] Pattern matching
- [ ] Generics
- [ ] Trait system
- [ ] Module system

## Design Philosophy

1. **Immutability First**: Easier to reason about programs, better for parallelization
2. **Flexibility**: Let users define types, not the language
3. **Pure by Default**: Functions are pure unless explicitly marked otherwise
4. **Type Safety**: Strong type checking without verbosity

## Extending the Language

### Adding a New Type

Types in CustomLang are user-defined identifiers. No changes to the grammar needed - just use them:

```customlang
let my_value: MyCustomType = ...;
let arr: MyCustomType[5] = ...;
let ptr: &MyCustomType = ...;
```

### Adding a New Operation

1. Update `grammar/CustomLangParser.g4` with new expression syntax
2. Define operation in MLIR tablegen file
3. Implement in `src/Ops.cpp`
4. Add visitor method in `ASTBuilder`

### Adding a New Built-in Function

Modify the semantic analyzer to recognize built-in functions and lower them to appropriate MLIR operations.

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [ANTLR Documentation](https://www.antlr.org/)
- [LLVM Project](https://llvm.org/)
- [Immutable Data Structures](https://en.wikipedia.org/wiki/Persistent_data_structure)
