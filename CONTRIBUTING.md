# Contributing to Zenith

Thank you for your interest in contributing to Zenith! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/zenith.git
   cd zenith
   ```
3. **Set up the development environment** (see README.md)

## Development Workflow

### Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** following the coding standards

3. **Test your changes**:
   ```bash
   cd build
   ninja check-zenith
   ```

4. **Commit your changes** with a descriptive message:
   ```bash
   git commit -m "Add feature: description of feature"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Open a Pull Request** on GitHub

## Coding Standards

### C++ Code Style

- Follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- Use C++20 features where appropriate
- Include header guards in all header files
- Add comments for complex logic
- Keep functions small and focused

### File Organization

- Header files go in `include/Zenith/`
- Implementation files go in `lib/`
- Tests go in `test/`
- Examples go in `examples/`

### Naming Conventions

- **Types/Classes**: `PascalCase` (e.g., `ConstantOp`, `ZenithDialect`)
- **Functions/Methods**: `camelCase` (e.g., `runOnOperation`, `getFunctionType`)
- **Variables**: `camelCase` (e.g., `resultType`, `operandValue`)
- **Constants**: `kPascalCase` (e.g., `kMaxInlineSize`)

### Documentation

- Add doxygen comments to all public APIs
- Update documentation in `docs/` for new features
- Include examples in documentation

Example:
```cpp
/// Adds two values together.
///
/// This operation performs element-wise addition between two values
/// of the same type.
///
/// Example:
/// ```mlir
/// %sum = zenith.add %a, %b : i32
/// ```
class AddOp : public Op<AddOp, ...> {
  // ...
};
```

## Testing

### Writing Tests

- Add lit tests for new operations and passes
- Test both positive and negative cases
- Include edge cases in tests

Example test:
```mlir
// RUN: zenith-opt %s | FileCheck %s

// CHECK-LABEL: func @test_add
func.func @test_add(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: zenith.add
    %0 = zenith.add %arg0, %arg1 : i32
    return %0 : i32
}
```

### Running Tests

```bash
cd build
ninja check-zenith           # Run all tests
ninja check-zenith-dialect   # Run dialect tests only
```

## Adding New Features

### Adding a New Operation

1. Define the operation in TableGen (`ZenithOps.td`)
2. Implement any required methods in C++ (`ZenithOps.cpp`)
3. Add tests for the operation
4. Update documentation

### Adding a New Pass

1. Define the pass in TableGen (`Passes.td`)
2. Implement the pass in C++ (`lib/Passes/`)
3. Register the pass in `Passes.h`
4. Add tests for the pass
5. Update documentation

### Adding a New Type

1. Define the type in `ZenithTypes.h`
2. Implement the type in `ZenithTypes.cpp`
3. Register the type in `ZenithDialect.cpp`
4. Add tests and documentation

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows the style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are descriptive
- [ ] No merge conflicts with main branch

### PR Description

Include in your PR description:
- **What**: Brief description of changes
- **Why**: Motivation for the changes
- **How**: Technical details of implementation
- **Testing**: How the changes were tested

Example:
```
## Add constant folding for multiplication

### What
Implements constant folding for the zenith.mul operation.

### Why
Enables compile-time evaluation of constant multiplications,
reducing runtime overhead.

### How
Added fold() method to MulOp that checks if both operands
are constants and returns the computed result.

### Testing
Added lit tests in test/Dialect/constant-fold.mlir
All existing tests still pass.
```

## Code Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, a maintainer will merge your PR

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat (link in README)

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## License

By contributing to Zenith, you agree that your contributions will be licensed under the Apache 2.0 License with LLVM Exceptions.

## Questions?

If you have questions about contributing, feel free to:
- Open an issue on GitHub
- Ask in GitHub Discussions
- Reach out on Discord

Thank you for contributing to Zenith! 🚀

