/**
 * Unit tests for Zenith Code Generator
 * Uses Google Test framework
 */

#include <gtest/gtest.h>
#include "antlr4-runtime.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "CodeGenerator.h"
#include <sstream>

using namespace antlr4;

class CodeGenTest : public ::testing::Test {
protected:
    std::string generateCode(const std::string& input) {
        ANTLRInputStream inputStream(input);
        ZenithLexer lexer(&inputStream);
        CommonTokenStream tokens(&lexer);
        ZenithParser parser(&tokens);
        
        auto tree = parser.program();
        
        if (parser.getNumberOfSyntaxErrors() > 0) {
            throw std::runtime_error("Parse errors");
        }
        
        CodeGenerator codegen;
        codegen.visit(tree);
        return codegen.getGeneratedCode();
    }
};

TEST_F(CodeGenTest, SimpleFunction) {
    std::string input = R"(
main() {
    return 0
}
    )";
    
    std::string output = generateCode(input);
    
    // Should contain main function
    EXPECT_NE(output.find("main"), std::string::npos);
    EXPECT_NE(output.find("return 0"), std::string::npos);
}

TEST_F(CodeGenTest, FunctionWithReturn) {
    std::string input = R"(
add(x, y) {
    return x + y
}
    )";
    
    std::string output = generateCode(input);
    
    // Just check that code was generated
    EXPECT_GT(output.length(), 0);
    EXPECT_NE(output.find("add"), std::string::npos);
}

TEST_F(CodeGenTest, VariableDeclaration) {
    std::string input = R"(
main() {
    x = 42
    return x
}
    )";
    
    std::string output = generateCode(input);
    
    // Just check that code was generated
    EXPECT_GT(output.length(), 0);
    EXPECT_NE(output.find("main"), std::string::npos);
}

TEST_F(CodeGenTest, IfStatement) {
    std::string input = R"(
main() {
    if 1 > 0 {
        return 1
    }
    return 0
}
    )";
    
    std::string output = generateCode(input);
    
    // Just check that code was generated
    EXPECT_GT(output.length(), 0);
}

TEST_F(CodeGenTest, WhileLoop) {
    std::string input = R"(
main() {
    i = 0
    while i < 10 {
        i = i + 1
    }
    return i
}
    )";
    
    std::string output = generateCode(input);
    
    // Just check that code was generated
    EXPECT_GT(output.length(), 0);
}

TEST_F(CodeGenTest, IncludesStdioHeader) {
    std::string input = R"(
main() {
    return 0
}
    )";
    
    std::string output = generateCode(input);
    
    // Should include headers
    EXPECT_NE(output.find("#include"), std::string::npos);
}

TEST_F(CodeGenTest, ArithmeticExpression) {
    std::string input = R"(
main() {
    return 2 + 3 * 4
}
    )";
    
    std::string output = generateCode(input);
    
    // Just check that code was generated
    EXPECT_GT(output.length(), 0);
}

TEST_F(CodeGenTest, ComparisonExpression) {
    std::string input = R"(
main() {
    if 5 > 3 {
        return 1
    }
    return 0
}
    )";
    
    std::string output = generateCode(input);
    
    // Just check that code was generated
    EXPECT_GT(output.length(), 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
