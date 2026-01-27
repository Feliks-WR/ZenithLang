# Testing Guide

## Overview

The Zenith compiler uses **Google Test** for unit and integration testing, integrated with CMake/CTest for automated testing.

## Test Structure

### Unit Tests
- **Parser Tests** (`tests/test_parser.cpp`) - Tests ANTLR grammar and parser functionality
- **Code Generator Tests** (`tests/test_codegen.cpp`) - Tests C code generation

### Integration Tests
- **End-to-End Tests** (`tests/integration_test.cpp`) - Tests complete compilation pipeline

## Running Tests

### Quick Start
```bash
# Build with tests
cmake -B build -G Ninja
cmake --build build

# Run all tests
cd build
ctest --output-on-failure

# Run with verbose output
ctest --output-on-failure --verbose
```

### Run Specific Test Suites
```bash
cd build

# Parser tests only
./test_parser

# Code generation tests only
./test_codegen

# Integration tests only
./integration_test
```

### Run Individual Tests
```bash
cd build

# Run specific test by name
./test_parser --gtest_filter=ParserTest.SimpleFunctionDeclaration

# List all available tests
./test_parser --gtest_list_tests
```

## Installing Google Test

### Ubuntu/Debian
```bash
sudo apt-get install libgtest-dev
```

### macOS
```bash
brew install googletest
```

### From Source
```bash
git clone https://github.com/google/googletest.git
cd googletest
cmake -B build
cmake --build build
sudo cmake --install build
```

## Writing New Tests

### Parser Test Example
```cpp
TEST_F(ParserTest, YourNewTest) {
    std::string input = R"(
        fn example() -> int {
            return 42;
        }
    )";
    
    EXPECT_FALSE(hasErrors(input));
    auto tree = parse(input);
    EXPECT_NE(tree, nullptr);
}
```

### Code Generation Test Example
```cpp
TEST_F(CodeGenTest, YourNewTest) {
    std::string input = R"(
        fn example() -> int {
            return 42;
        }
    )";
    
    std::string output = generateCode(input);
    EXPECT_NE(output.find("return 42"), std::string::npos);
}
```

### Integration Test Example
```cpp
TEST_F(IntegrationTest, YourNewTest) {
    std::string source = R"(
        fn main() -> int {
            return 42;
        }
    )";
    
    auto [success, output] = compileAndRun(source, "test_name");
    EXPECT_TRUE(success) << "Failed: " << output;
}
```

## Test Coverage

Current test coverage includes:
- ✅ Parser validation
- ✅ Function declarations
- ✅ Variable declarations
- ✅ Control flow (if/while)
- ✅ Arithmetic expressions
- ✅ Comparison operators
- ✅ Code generation
- ✅ End-to-end compilation

## CI/CD Pipeline

Tests run automatically on:
- Push to `master`, `main`, or `develop` branches
- Pull requests to these branches

GitHub Actions workflow includes:
1. Dependency installation
2. CMake configuration
3. Build compilation
4. Test execution
5. Artifact upload

View CI results at: https://github.com/Feliks-WR/ZenithLang/actions

## Debugging Test Failures

### Verbose output
```bash
cd build
ctest --output-on-failure --verbose
```

### Run with GDB
```bash
cd build
gdb ./test_parser
(gdb) run
```

### Check compiler output
```bash
./zenith tests/test_basic.zenith --emit-c
cat test_basic.c
```

## Best Practices

1. **Test names** - Use descriptive names that explain what is being tested
2. **Assertions** - Use appropriate Google Test assertions (EXPECT_*, ASSERT_*)
3. **Isolation** - Each test should be independent
4. **Coverage** - Test both success and failure cases
5. **Documentation** - Add comments for complex test logic

## Continuous Integration

The CI pipeline ensures:
- ✅ Code builds on clean Ubuntu environment
- ✅ All tests pass
- ✅ Compiler produces working binaries
- ✅ Examples compile successfully

## Troubleshooting

### "GTest not found"
```bash
sudo apt-get install libgtest-dev
cmake -B build
```

### "antlr4-runtime not found"
```bash
sudo apt-get install libantlr4-runtime-dev
```

### Test timeout
Increase timeout in CMakeLists.txt:
```cmake
set_tests_properties(TestName PROPERTIES TIMEOUT 30)
```

## Resources

- [Google Test Documentation](https://google.github.io/googletest/)
- [CMake Testing](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
- [GitHub Actions](https://docs.github.com/en/actions)
