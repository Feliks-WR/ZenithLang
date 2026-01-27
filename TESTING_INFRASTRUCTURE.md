# Zenith Language - Testing Infrastructure

## ✅ Complete Testing Pipeline

### Test Framework: Google Test
- Industry-standard C++ testing framework
- Rich assertion library
- Test fixtures and parameterized tests
- Clear, readable test output

### Test Suite Structure

#### 1. **Unit Tests** (tests/test_parser.cpp)
- Parser validation
- Grammar correctness
- Syntax error detection
- Token recognition
- **11 test cases** covering:
  - Empty programs
  - Function declarations
  - Parameters
  - Variables
  - Control flow
  - Expressions
  - Invalid syntax

#### 2. **Code Generation Tests** (tests/test_codegen.cpp)
- C code output validation
- Correct translation
- Header inclusion
- Type conversion
- **8 test cases** covering:
  - Simple functions
  - Function parameters
  - Variable declarations
  - Control structures
  - Expressions
  - Header generation

#### 3. **Integration Tests** (tests/integration_test.cpp)
- End-to-end compilation
- Binary execution
- Runtime behavior
- **7 test cases** covering:
  - Simple returns
  - Arithmetic operations
  - Conditionals
  - Loops
  - Function calls
  - Nested constructs
  - Complex expressions

#### 4. **Example Programs** (.zenith files)
- test_basic.zenith - Basic syntax
- test_comprehensive.zenith - Multiple features
- test_nested.zenith - Nested control flow
- test_functions.zenith - Function calls
- test_invalid_syntax.zenith - Error handling

**Total: 26+ automated test cases**

## 🔄 CI/CD Pipeline

### Main CI Workflow (.github/workflows/ci.yml)
**Triggered on:** Push to master/main/develop, Pull Requests

**Jobs:**
1. **build-and-test**
   - Install dependencies (GTest, ANTLR4, CMake)
   - Configure with CMake
   - Build with Ninja
   - Run all tests with CTest
   - Upload compiler artifact
   
2. **test-examples**
   - Download built compiler
   - Test example programs
   - Validate compilation
   
3. **code-quality**
   - Check file formatting
   - Validate grammar files
   - Source code verification

### Coverage Workflow (.github/workflows/coverage.yml)
**Triggered on:** Push to master/main, Weekly schedule

- Build with coverage flags
- Run all tests
- Generate LCOV report
- Upload to Codecov

### Release Workflow (.github/workflows/release.yml)
**Triggered on:** Version tags (v*), Manual dispatch

- Build on Ubuntu and macOS
- Run full test suite
- Package binaries
- Create release artifacts

## 📊 Running Tests Locally

### Quick Start
```bash
# Option 1: Using Make
make build
make test

# Option 2: Using CMake directly
cmake -B build -G Ninja
cmake --build build
cd build && ctest --output-on-failure

# Option 3: Individual test suites
cd build
./test_parser
./test_codegen
./integration_test
```

### Verbose Output
```bash
cd build
ctest --output-on-failure --verbose
```

### Specific Tests
```bash
cd build
./test_parser --gtest_filter=ParserTest.SimpleFunctionDeclaration
./test_codegen --gtest_list_tests
```

## 🛠️ Development Tools

### Makefile Targets
- `make build` - Build compiler
- `make test` - Run tests
- `make run-tests` - Verbose test output
- `make clean` - Clean artifacts
- `make install-deps` - Install dependencies
- `make format` - Format code
- `make ci-test` - Full CI test locally

### CTest Configuration
- Test timeout: 30 seconds
- Parallel execution: 4 threads
- Output on failure: Enabled
- Verbose mode available

## 📈 Test Coverage

### Current Coverage
- ✅ Lexer/Parser (ANTLR grammar)
- ✅ Code generation (C output)
- ✅ Variable declarations
- ✅ Function declarations
- ✅ Control flow (if/while)
- ✅ Expressions (arithmetic, comparison)
- ✅ End-to-end compilation
- ✅ Runtime execution
- ✅ Error handling

### Quality Metrics
- **Unit Tests:** 19 tests
- **Integration Tests:** 7 tests
- **Example Programs:** 5 files
- **CI/CD Workflows:** 3 pipelines
- **Build Systems:** CMake + Make
- **Platforms:** Ubuntu, macOS
- **Test Framework:** Google Test

## 🚀 Continuous Integration Benefits

1. **Automated Testing** - Every push runs full test suite
2. **Cross-Platform** - Validates Ubuntu and macOS
3. **Quality Gates** - PR must pass tests before merge
4. **Artifact Generation** - Downloadable binaries
5. **Coverage Tracking** - Code coverage reports
6. **Release Automation** - Tagged releases build automatically

## 📝 Adding New Tests

### Parser Test Template
```cpp
TEST_F(ParserTest, YourFeature) {
    std::string input = R"(
        // Your Zenith code
    )";
    EXPECT_FALSE(hasErrors(input));
    auto tree = parse(input);
    EXPECT_NE(tree, nullptr);
}
```

### Code Gen Test Template
```cpp
TEST_F(CodeGenTest, YourFeature) {
    std::string input = R"(
        // Your Zenith code
    )";
    std::string output = generateCode(input);
    EXPECT_NE(output.find("expected string"), std::string::npos);
}
```

### Integration Test Template
```cpp
TEST_F(IntegrationTest, YourFeature) {
    std::string source = R"(
        fn main() -> int {
            // Your code
            return 0;
        }
    )";
    auto [success, output] = compileAndRun(source, "test_name");
    EXPECT_TRUE(success) << "Failed: " << output;
}
```

## 🎯 Best Practices

1. **Test Naming** - Descriptive, explains what's tested
2. **Independence** - Each test isolated, no dependencies
3. **Coverage** - Test success AND failure cases
4. **Documentation** - Comments for complex logic
5. **Assertions** - Use appropriate EXPECT_/ASSERT_ macros
6. **Fast Tests** - Keep execution time minimal
7. **CI First** - Ensure tests pass in CI environment

## 📚 Resources

- [Google Test Primer](https://google.github.io/googletest/primer.html)
- [CMake Testing Guide](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Zenith Testing Guide](docs/TESTING.md)

## 🎉 Summary

The Zenith compiler now has a **production-grade testing infrastructure**:

✅ **26+ automated tests** across unit, integration, and e2e  
✅ **Google Test framework** for professional C++ testing  
✅ **CMake/CTest integration** for build system testing  
✅ **3 GitHub Actions workflows** (CI, Coverage, Release)  
✅ **Cross-platform support** (Ubuntu, macOS)  
✅ **Make targets** for convenient development  
✅ **Comprehensive documentation** in docs/TESTING.md  
✅ **CI badges** for repository status  
✅ **Artifact generation** for downloads  
✅ **Code coverage** tracking

**No Python scripts, just industry-standard tools!**
