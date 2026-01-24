#include "antlr4-runtime.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "ZenithParserBaseVisitor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// MLIR / LLVM are required dependencies for this project (see docs/AI_INSTRUCTIONS.md)
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "ASTBuilder.h"

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
        if (ctx->identifierList()) {
            for (auto id : ctx->identifierList()->IDENTIFIER()) {
                std::cout << id->getText() << " ";
            }
        }
        if (ctx->type()) {
            std::cout << ": " << ctx->type()->getText();
        }
        std::cout << "\n";
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
        if (ctx->blockStatement().size() > 0) visit(ctx->blockStatement(0));
        if (ctx->ELSE()) {
            std::cout << "Else:\n";
            if (ctx->blockStatement().size() > 1) visit(ctx->blockStatement(1));
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
        std::cerr << "Usage: zenith <file> [--emit-llvm|--jit]" << std::endl;
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

    // Visit AST and show parse info
    ZenithVisitor visitor;
    visitor.visit(tree);

#ifdef USE_MLIR
    mlir::MLIRContext context;
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();
    context.appendDialectRegistry(registry);

    // Build MLIR module from AST
    ASTBuilder astBuilder(&context);
    astBuilder.visit(tree);
    auto module = astBuilder.getModule();

    if (!module) {
        std::cerr << "Failed to build MLIR module\n";
        return 1;
    }

    std::cout << "--- MLIR Module ---\n";
    module->print(llvm::outs());
    std::cout << "\n--- end MLIR ---\n";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--emit-llvm") {
            llvm::LLVMContext llvmContext;
            auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
            if (!llvmModule) {
                std::cerr << "Failed to translate to LLVM IR\n";
                return 1;
            }
            llvmModule->print(llvm::outs(), nullptr);
            return 0;
        }
        if (arg == "--jit") {
            auto maybeEngine = mlir::ExecutionEngine::create(*module);
            if (!maybeEngine) {
                llvm::errs() << "Failed to create ExecutionEngine\n";
                return 1;
            }
            auto &engine = *maybeEngine;
            int (*fn)();
            if (engine.lookup("main", (void **)&fn)) {
                llvm::errs() << "Lookup failed\n";
                return 1;
            }
            int result = fn();
            std::cout << "JIT result: " << result << "\n";
            return 0;
        }
    }
#else
    (void)argc; (void)argv; // silence unused warnings
    std::cout << "MLIR not available in this build; rebuild with MLIR to enable emit/jit modes.\n";
#endif

    std::cout << "✓ Parsed successfully\n";
    return 0;
}
