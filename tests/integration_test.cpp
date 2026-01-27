/**
 * Integration tests for Zenith Compiler
 * Tests end-to-end compilation and execution
 */

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

class IntegrationTest : public ::testing::Test {
protected:
    fs::path test_dir;
    std::string compiler_path;
    
    void SetUp() override {
        test_dir = fs::temp_directory_path() / "zenith_test";
        fs::create_directories(test_dir);
        
        // Find compiler executable
        if (fs::exists("./zenith")) {
            compiler_path = "./zenith";
        } else if (fs::exists("../zenith")) {
            compiler_path = "../zenith";
        } else if (fs::exists("./build/zenith")) {
            compiler_path = "./build/zenith";
        } else {
            FAIL() << "Compiler executable not found";
        }
    }
    
    void TearDown() override {
        fs::remove_all(test_dir);
    }
    
    std::pair<bool, std::string> compileAndRun(const std::string& source, const std::string& name = "test") {
        fs::path source_file = test_dir / (name + ".zenith");
        fs::path output_file = test_dir / name;
        
        // Write source file
        {
            std::ofstream out(source_file);
            out << source;
        }
        
        // Compile
        std::string compile_cmd = compiler_path + " " + source_file.string() + " -o " + output_file.string() + " 2>&1";
        FILE* pipe = popen(compile_cmd.c_str(), "r");
        if (!pipe) return {false, "Failed to execute compiler"};
        
        std::string compile_output;
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            compile_output += buffer;
        }
        int compile_status = pclose(pipe);
        
        if (compile_status != 0) {
            return {false, "Compilation failed: " + compile_output};
        }
        
        // Run
        std::string run_cmd = output_file.string() + " 2>&1";
        pipe = popen(run_cmd.c_str(), "r");
        if (!pipe) return {false, "Failed to execute binary"};
        
        std::string run_output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            run_output += buffer;
        }
        pclose(pipe);
        
        return {true, run_output};
    }
};

TEST_F(IntegrationTest, SimpleReturn) {
    std::string source = R"(
main() {
    return 0
}
    )";
    
    auto [success, output] = compileAndRun(source, "simple_return");
    EXPECT_TRUE(success) << "Compilation/execution failed: " << output;
}

TEST_F(IntegrationTest, ArithmeticOperation) {
    std::string source = R"(
main() {
    x = 10
    y = 5
    result = x + y
    return result
}
    )";
    
    auto [success, output] = compileAndRun(source, "arithmetic");
    EXPECT_TRUE(success) << "Compilation/execution failed: " << output;
}

TEST_F(IntegrationTest, ConditionalBranch) {
    std::string source = R"(
main() {
    x = 10
    if x > 5 {
        return 1
    }
    return 0
}
    )";
    
    auto [success, output] = compileAndRun(source, "conditional");
    EXPECT_TRUE(success) << "Compilation/execution failed: " << output;
}

TEST_F(IntegrationTest, WhileLoopCounter) {
    std::string source = R"(
main() {
    i = 0
    sum = 0
    while i < 5 {
        sum = sum + i
        i = i + 1
    }
    return sum
}
    )";
    
    auto [success, output] = compileAndRun(source, "while_loop");
    EXPECT_TRUE(success) << "Compilation/execution failed: " << output;
}

TEST_F(IntegrationTest, FunctionCall) {
    std::string source = R"(
add(a, b) {
    return a + b
}

main() {
    result = add(5, 3)
    return result
}
    )";
    
    auto [success, output] = compileAndRun(source, "function_call");
    EXPECT_TRUE(success) << "Compilation/execution failed: " << output;
}

TEST_F(IntegrationTest, NestedConditions) {
    std::string source = R"(
main() {
    x = 10
    y = 20
    
    if x < y {
        if y > 15 {
            return 1
        }
        return 2
    }
    return 0
}
    )";
    
    auto [success, output] = compileAndRun(source, "nested_conditions");
    EXPECT_TRUE(success) << "Compilation/execution failed: " << output;
}

TEST_F(IntegrationTest, ComplexArithmetic) {
    std::string source = R"(
main() {
    a = 2
    b = 3
    c = 4
    result = (a + b) * c - 5
    return result
}
    )";
    
    auto [success, output] = compileAndRun(source, "complex_arithmetic");
    EXPECT_TRUE(success) << "Compilation/execution failed: " << output;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
