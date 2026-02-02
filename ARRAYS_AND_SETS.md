# Arrays and Sets Support in Zenith

## Overview

This document describes the comprehensive support for arrays and sets in the Zenith language's type system, including dependent types and constraints.

## Implementation

### Type System Extensions

#### 1. **Array Support** (Previously Implemented)

- **Syntax**: `[ElementType; N]`
- **Example**: `[int; 10]` - array of 10 integers
- **Type Definition**: `DependentType::Array`
- **Factory Method**: `DependentType::makeArray(elemType, lengthParam)`

#### 2. **Set Support** (Newly Added)

- **Type Definition**: `DependentType::Set` - added to `TypeKind` enum
- **Storage**: `setElementType` field holds the element type
- **Factory Method**: `DependentType::makeSet(elemType)`
- **String Representation**: `{ElementType}` format

### Files Modified

#### [include/Types.h](include/Types.h)

- Added `Set` to `TypeKind` enum (line 54)
- Added `setElementType` field to store the set element type (line 71)
- Added `makeSet()` factory method declaration (lines 85-86)

#### [src/Types.cpp](src/Types.cpp)

- Implemented `DependentType::makeSet()` factory function (lines 88-92)
- Added `case Set:` to `toString()` method (lines 143-150)
  - Renders as `{ElementType}` in string form
- Added set type compatibility checking in `isCompatibleWith()` (lines 204-210)
  - Ensures element types are compatible

### Features

#### Type String Representation

```cpp
// Array example
DependentType::makeArray(DependentType::makeInt(), "N")->toString()
// Output: "[int; N]"

// Set example
DependentType::makeSet(DependentType::makeInt())->toString()
// Output: "{int}"
```

#### Type Compatibility

Both arrays and sets support compatibility checking:

- Element types must match
- For arrays, length parameters must also match
- Sets only require element type compatibility

#### Constraint Support

Both array and set types inherit full constraint support from the dependent type system:

```cpp
// Examples (planned grammar support)
// Constrained array: [int; N] {N > 0}
// Constrained set: {int} {size > 5}
```

## Usage Examples

### Creating Set Types Programmatically

```cpp
// Create a set of integers
auto intSet = DependentType::makeSet(DependentType::makeInt());

// Create a set with constraints
auto constrainedSet = DependentType::makeSet(DependentType::makeFloat());
constrainedSet->constraints.push_back(
    Constraint::makePredicate("nonempty")
);
```

### Type Checking

```cpp
auto set1 = DependentType::makeSet(DependentType::makeInt());
auto set2 = DependentType::makeSet(DependentType::makeInt());

if (set1->isCompatibleWith(set2)) {
    // Types are compatible
}
```

## Future Enhancements

1. **Grammar Support**: Add set and array literals to ZenithParser.g4
   - Array literals: `[1, 2, 3]`
   - Set literals: `{1, 2, 3}`

2. **CodeGen Support**: Generate C code for array/set operations
   - Array indexing: `arr[i]`
   - Set operations: insertion, removal, membership testing

3. **Runtime Support**: Implement array/set data structures in generated code
   - C arrays for fixed-size arrays
   - Hash tables or trees for sets

4. **Type Operations**: Built-in functions
   - `array.length()`, `array.size()`
   - `set.contains()`, `set.insert()`, `set.remove()`

## Testing

### Unit Tests

The implementation is tested through the existing `test_codegen` and `test_parser` test suites. To add specific tests for arrays and sets:

```cpp
TEST(TypesTest, ArrayCreation) {
    auto intArray = DependentType::makeArray(
        DependentType::makeInt(), "N"
    );
    EXPECT_EQ(intArray->kind, DependentType::Array);
}

TEST(TypesTest, SetCreation) {
    auto intSet = DependentType::makeSet(
        DependentType::makeInt()
    );
    EXPECT_EQ(intSet->kind, DependentType::Set);
}
```

## Architecture

The implementation follows the existing Zenith architecture:

1. **Type Definition**: In `include/Types.h` as part of `DependentType` class
2. **Factory Methods**: Static creation functions for safe shared_ptr management
3. **Visitor Pattern**: Compatible with existing code generator visitor
4. **Constraint System**: Inherits full dependent type constraint checking

## Notes

- Arrays support length parameters for dependent typing
- Sets use homogeneous element types
- Both types compose with pointers and other types
- Compatible with the existing constraint system for verification
