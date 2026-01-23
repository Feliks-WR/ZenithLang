#include "antlr4-runtime.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "ZenithParserBaseVisitor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace antlr4;

class ZenithVisitor : public ZenithParserBaseVisitor {
public:
    std::any visitProgram(ZenithParser::ProgramContext *ctx) override {
        std::cout << "Program with " << ctx->statement().size() << " statements\n";
        for (auto stmt : ctx->statement()) {
            visit(stmt);
        }
        return nullptr;
    }

    std::any visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) override {
        std::cout << "Declaration: ";
        for (auto id : ctx->identifierList()->IDENTIFIER()) {
            std::cout << id->getText() << " ";
        }
        std::cout << ": " << ctx->type()->getText() << "\n";
        return nullptr;
    }

    std::any visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) override {
        std::string name = ctx->IDENTIFIER()->getText();
        std::cout << "Function: " << name;
        if (ctx->ARROW()) {
            std::cout << " -> " << ctx->type()->getText();
        }
        std::cout << "\n";
        return nullptr;
    }

    std::any visitEquation(ZenithParser::EquationContext *ctx) override {
        std::cout << "Equation: " << ctx->expression(0)->getText() << " = " << ctx->expression(1)->getText() << "\n";
        return nullptr;
    }

    std::any visitExprStatement(ZenithParser::ExprStatementContext *ctx) override {
        std::cout << "Expression: " << ctx->expression()->getText() << "\n";
        return nullptr;
    }

    std::any visitCallExpr(ZenithParser::CallExprContext *ctx) override {
        auto primary = ctx->primaryExpr()->getText();
        auto suffixes = ctx->callSuffix();
        
        if (!suffixes.empty()) {
            std::cout << "  Call: " << primary << " with " << suffixes.size() << " args\n";
        }
        
        return visitChildren(ctx);
    }

    std::any visitIfStatement(ZenithParser::IfStatementContext *ctx) override {
        std::cout << "If: " << ctx->expression()->getText() << "\n";
        visit(ctx->blockStatement(0));
        if (ctx->ELSE()) {
            std::cout << "Else:\n";
            visit(ctx->blockStatement(1));
        }
        return nullptr;
    }

    std::any visitBlockStatement(ZenithParser::BlockStatementContext *ctx) override {
        for (auto stmt : ctx->statement()) {
            visit(stmt);
        }
        return nullptr;
    }
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: zenith <file>\n";
        return 1;
    }

    // Read file
    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Error: cannot open file '" << argv[1] << "'\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    // Parse
    ANTLRInputStream input(source);
    ZenithLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    ZenithParser parser(&tokens);

    // Get AST
    ZenithParser::ProgramContext *tree = parser.program();

    if (parser.getNumberOfSyntaxErrors() > 0) {
        std::cerr << "Parse errors detected\n";
        return 1;
    }

    // Visit AST
    ZenithVisitor visitor;
    visitor.visit(tree);

    std::cout << "✓ Parsed successfully\n";
    return 0;
}
