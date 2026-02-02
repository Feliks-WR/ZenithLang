<<<<<<< HEAD
# Dependent Types System for Zenith Language

## Overview

The Zenith language now supports **dependent types**, a powerful feature allowing types to depend on values. This enables compile-time verification of safety properties like:

- Array bounds checking
- Division by zero prevention
- Pointer validity
- Custom value constraints

## Syntax

### Basic Types with Constraints

#### Integer Constraints

**Range constraint:**

```zenith
age : int {1..120}
```

**Predicate constraint (implicit):**

```zenith
divisor : int {(!=0)}
```

**Predicate constraint (explicit with `it`):**

```zenith
divisor : int {it != 0}
```

#### Pointer Types

**Basic pointer:**

```zenith
ptr : *int
```

**Non-null pointer:**

```zenith
validPtr : *int {nonnull}
```

**Pointer binding with destructuring:**

```zenith
*x = 5  // x is bound to the dereferenced value
```

#### Array Types

**Parameterized array (length parameter):**

```zenith
data : [int; N]  // N is a type parameter
buffer : [float; 100]  // Fixed-size array
```

## Type System Components

### 1. Constraint Class

Represents compile-time proof obligations.

**Kinds:**

- `Range` - Integer range (e.g., `{1..10}`)
- `SingleValue` - Specific value (e.g., `{5}`)
- `Predicate` - Boolean predicate (e.g., `{it != 0}`)
- `NonNull` - Pointer non-null (e.g., `{nonnull}`)
- `Custom` - User-defined constraints

**Example:**

```cpp
auto constraint = Constraint::makeRange(1, 10);
auto predicate = Constraint::makePredicate("it != 0");
auto nonnull = Constraint::makeNonNull();
```

### 2. DependentType Class

Represents types that may depend on values.

**Kinds:**

- `Int` - Integer type
- `Float` - Floating-point type
- `Bool` - Boolean type
- `Pointer` - Pointer type (carries element type)
- `Array` - Array type (carries element type and length parameter)
- `Function` - Function type
- `Named` - Custom/named type

**Example:**

```cpp
// Integer with non-zero constraint
auto divisor = DependentType::makeIntWithConstraint(
    Constraint::makePredicate("it != 0"));

// Array of 10 integers
auto arr = DependentType::makeArray(
    DependentType::makeInt(), "10");

// Non-null pointer to int
auto ptr = DependentType::makePointer(DependentType::makeInt());
ptr->constraints.push_back(Constraint::makeNonNull());
```

### 3. TypeChecker Class

Enforces compile-time type constraints and collects proof obligations.

**Key Methods:**

- `declareVariable(name, type)` - Register variable with dependent type
- `assignVariable(name, value)` - Check value satisfies type constraints
- `checkArrayAccess(arrayType, index, location)` - Verify array bounds
- `checkPointerDereference(ptrType, location)` - Verify pointer validity
- `checkDivision(divisorType, expr, location)` - Verify non-zero divisor
- `checkModulo(divisorType, expr, location)` - Verify non-zero modulo divisor
- `getUnsatisfiedObligations()` - Get remaining proof obligations
- `allObligationsSatisfied()` - Check if all proofs are satisfied

## Proof Obligations

The type checker generates proof obligations for potentially unsafe operations:

### ProofObligation Types

```cpp
enum Kind {
    ArrayBounds,       // arr[i] - i must be < length
    DivisionNonZero,   // x / y - y must != 0
    ModuloNonZero,     // x % y - y must != 0
    PointerDeref,      // *p - p must be non-null
    PointerValid,      // pointer arithmetic validity
    Custom             // user-defined
};
```

## Constraint Satisfaction

The system verifies that values satisfy type constraints:

```cpp
// Check if value satisfies constraints
bool satisfies = type->satisfiesConstraints("5");

// For predicates like "it != 0"
bool valid = divisor->satisfiesConstraints("3");  // true
bool invalid = divisor->satisfiesConstraints("0");  // false

// For ranges
bool inRange = age->satisfiesConstraints("25");  // true if 1..120
bool outOfRange = age->satisfiesConstraints("150");  // false
```

## Examples

### Safe Division

```zenith
// Define a type for non-zero integers
divisor : int {it != 0}
divisor = 5

// Now we can safely divide
x = 10
result = x / divisor  // Type checker verified divisor != 0
```

### Bounded Arrays

```zenith
// Array of fixed size
scores : [int; 100]

// Access with bounds check
index : int {0..99}
score = scores[index]  // Proven safe: 0 <= index < 100

// Safe pointer access
ptr : *int {nonnull}
val = *ptr  // Safe: ptr is guaranteed non-null
```

### Type-Value Dependencies

```zenith
// Day must be in valid range
day : int {1..31}
day = 15

// Array parameterized by type parameter
buffer : [int; N]  // N is a length parameter

// Pointer binding
node : *int
*x = 5  // x is now the dereferenced value
```

## Integration with Type Checker

```cpp
TypeChecker checker;

// Declare variables with dependent types
auto nonZeroInt = DependentType::makeIntWithConstraint(
    Constraint::makePredicate("it != 0"));
checker.declareVariable("divisor", nonZeroInt);

// Assign and verify constraints
checker.assignVariable("divisor", "5");  // OK
checker.assignVariable("divisor", "0");  // Error

// Check operations for proof obligations
auto arrayType = DependentType::makeArray(
    DependentType::makeInt(), "10");
checker.checkArrayAccess(arrayType, "i", "line_5");

// Retrieve unsatisfied proofs
auto obligations = checker.getUnsatisfiedObligations();
for (const auto& obligation : obligations) {
    std::cout << obligation.description << std::endl;
}
```

## Compilation Process

1. **Parsing**: Grammar parses dependent type syntax
2. **Type Checking**: Type checker registers types and verifies constraints
3. **Proof Collection**: Unsafe operations generate proof obligations
4. **Proof Verification**: Compiler verifies all obligations are satisfied
5. **Code Generation**: Generate safe code with proven invariants

## Testing

Comprehensive test suite included in `tests/test_dependent_types.cpp`:

- 23+ unit tests covering constraints, types, and type checking
- All tests passing
- Covers edge cases and error conditions

## Grammar Extensions

The parser grammar was extended to support dependent type syntax:

```antlr
type: dependentType;
dependentType: baseType constraint?;
baseType: IDENTIFIER | pointerType | arrayType;
pointerType: STAR baseType;
arrayType: LBRACKET baseType SEMICOLON IDENTIFIER RBRACKET;
constraint: LBRACE constraintExpr RBRACE;
```

## Future Enhancements

1. **Theorem Proving Integration**: Connect to SMT solvers for constraint verification
2. **Bidirectional Type Inference**: Infer dependent types from usage patterns
3. **Refinement Types**: Support more expressive constraints
4. **Dependent Function Types**: Functions with dependent arguments and return types
5. **Type-level Computation**: Enable compile-time value computation
=======
# Dependent Types in Zenith

Dependent types are now supported in the Zenith language! They allow you to attach predicates to types to express constraints and properties.

## Syntax

The basic syntax for dependent types is:

```
Type {predicate}
```

Where `Type` is any valid Zenith type and `predicate` is a constraint expression.

## Examples

### Simple Constraints

```zenith
// Non-zero integer
x: int {!= 0}

// Positive integer
y: int {> 0}

// Zero or positive
z: int {>= 0}
```

### Range Constraints

```zenith
// Value between 1 and 10
score: int {1..10}

// Value between 0 and 100
percentage: int {0..100}
```

### Keyword Constraints

Both `!` and `not` are supported for negation:

```zenith
// Non-empty string (using !)
name: str {!blank}

// Non-empty string (using not)
title: str {not blank}
```

### Function Parameters

```zenith
divide(a: int {!= 0}, b: int {> 0}) -> real {
    return a / b
}

getScore(x: int {0..100}) -> int {
    return x
}
```

### Function Return Types

```zenith
getPositive() -> int {> 0} {
    return 42
}
```

### Array Types

```zenith
// Array of integers (syntax in progress)
numbers: [int] {sorted}
```

## Predicate Types

### Infix Predicates

Used for comparing the value to a literal:

```
{!= 0}    // not equal to 0
{> 5}     // greater than 5
{<= 100}  // less than or equal to 100
{== 42}   // equal to 42
```

### Range Predicates

Used to constrain to a range:

```
{1..10}     // between 1 and 10
{0..100}    // between 0 and 100
{-5..5}     // between -5 and 5
```

### Unary Predicates

Applied to single operands:

```
{!x}        // not x
{not x}     // not x (keyword version)
{-1}        // negative one (literal)
```

### Complex Predicates

Named predicates (identifiers):

```
{sorted}            // is sorted
{blank}             // is blank
{custom_check}      // custom named predicate
```

## Limitations and Notes

- Dependent types are currently parsed but not enforced at compile time
- They serve as annotations for documentation and potential future runtime/compile-time checking
- The parser supports the syntax but code generation currently ignores the predicates
- Future versions may add compile-time verification or runtime assertions based on these predicates

## Test Examples

See `tests/test_dependent_types.zenith` for comprehensive examples.
>>>>>>> f44d684 (Add initial implementation of Zenith parser and visitor classes)
