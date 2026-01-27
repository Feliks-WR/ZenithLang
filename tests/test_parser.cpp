/**
 * Unit tests for Zenith Parser
 * Uses Google Test framework
 */

#include <gtest/gtest.h>
#include "antlr4-runtime.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include <sstream>

using namespace antlr4;

class ParserTest : public ::testing::Test {
protected:
    ZenithParser::ProgramContext* parse(const std::string& input) {
        ANTLRInputStream inputStream(input);
        ZenithLexer lexer(&inputStream);
        CommonTokenStream tokens(&lexer);
        ZenithParser parser(&tokens);
        return parser.program();
    }
    
    bool hasErrors(const std::string& input) {
        ANTLRInputStream inputStream(input);
        ZenithLexer lexer(&inputStream);
        CommonTokenStream tokens(&lexer);
        ZenithParser parser(&tokens);
        parser.program();
        return parser.getNumberOfSyntaxErrors() > 0;
    }
};

TEST_F(ParserTest, EmptyProgram) {
    EXPECT_NO_THROW({
        auto tree = parse("");
        EXPECT_NE(tree, nullptr);
    });
}

TEST_F(ParserTest, SimpleFunctionDeclaration) {
    std::string input = R"(
main() {
    return 0
}
    )";
    
    EXPECT_FALSE(hasErrors(input));
    auto tree = parse(input);
    EXPECT_NE(tree, nullptr);
    EXPECT_GT(tree->children.size(), 0);
}

TEST_F(ParserTest, FunctionWithParameters) {
    std::string input = R"(
add(x, y) {
    return x + y
}
    )";
    
    EXPECT_FALSE(hasErrors(input));
}

TEST_F(ParserTest, VariableDeclaration) {
    std::string input = R"(
main() {
    x = 42
    return x
}
    )";
    
    EXPECT_FALSE(hasErrors(input));
}

TEST_F(ParserTest, IfStatement) {
    std::string input = R"(
main() {
    x = 10
    if x > 5 {
        return 1
    }
    return 0
}
    )";
    
    EXPECT_FALSE(hasErrors(input));
}

TEST_F(ParserTest, WhileLoop) {
    std::string input = R"(
main() {
    i = 0
    while i < 10 {
        i = i + 1
    }
    return i
}
    )";
    
    EXPECT_FALSE(hasErrors(input));
}

TEST_F(ParserTest, ArithmeticExpression) {
    std::string input = R"(
main() {
    result = (2 + 3) * 4 - 1
    return result
}
    )";
    
    EXPECT_FALSE(hasErrors(input));
}

TEST_F(ParserTest, MultipleStatements) {
    std::string input = R"(
main() {
    a = 5
    b = 10
    c = a + b
    return c
}
    )";
    
    EXPECT_FALSE(hasErrors(input));
}

TEST_F(ParserTest, InvalidSyntaxShouldError) {
    std::string input = R"(
main( {
    return 0
}
    )";
    
    EXPECT_TRUE(hasErrors(input));
}

TEST_F(ParserTest, UnbalancedBraces) {
    std::string input = R"(
main() {
    return 0

    )";
    
    // Missing closing brace
    EXPECT_TRUE(hasErrors(input));
}

TEST_F(ParserTest, ComparisonOperators) {
    std::string input = R"(
main() {
    a = 5
    b = 10
    if a < b && b > a {
        return 1
    }
    return 0
}
    )";
    
    EXPECT_FALSE(hasErrors(input));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
