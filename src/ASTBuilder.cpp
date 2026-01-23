#include "ASTBuilder.h"
#include "mlir/IR/Module.h"

using namespace mlir;
using namespace mlir::customlang;

ASTBuilder::ASTBuilder(mlir::MLIRContext *context)
    : context(context), builder(context) {
  module = ModuleOp::create(UnknownLoc::get(context));
}

std::any ASTBuilder::visitProgram(CustomLangParser::ProgramContext *ctx) {
  // Visit all declarations in the program
  for (auto decl : ctx->declaration()) {
    visit(decl);
  }
  return nullptr;
}

std::any ASTBuilder::visitFunctionDecl(
    CustomLangParser::FunctionDeclContext *ctx) {
  // TODO: Implement function declaration handling
  return nullptr;
}

std::any ASTBuilder::visitVarDecl(CustomLangParser::VarDeclContext *ctx) {
  // TODO: Implement variable declaration handling
  return nullptr;
}

std::any ASTBuilder::visitExpression(CustomLangParser::ExpressionContext *ctx) {
  // TODO: Implement expression handling
  return nullptr;
}
