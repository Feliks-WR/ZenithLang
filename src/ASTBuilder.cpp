#include "ASTBuilder.h"
#ifdef USE_MLIR
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#endif

using namespace mlir::customlang;

#ifdef USE_MLIR
ASTBuilder::ASTBuilder(mlir::MLIRContext *context)
    : context(context), builder(context) {
  module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
}

std::any ASTBuilder::visitProgram(ZenithParser::ProgramContext *ctx) {
  // Visit top-level statements / declarations
  for (auto stmt : ctx->statement()) {
    visit(stmt);
  }
  return nullptr;
}

std::any ASTBuilder::visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) {
  // Create a function with i32 return for now
  auto i32Type = builder.getI32Type();
  auto funcType = builder.getFunctionType({}, {i32Type});
  std::string name = ctx->IDENTIFIER()->getText();
  auto func =
      mlir::func::FuncOp::create(builder.getUnknownLoc(), name, funcType);
  mlir::Block *entry = func.addEntryBlock();

  // Clear symbol table for new function
  symbolTable.clear();

  // Lower function body if present (blockStatement)
  auto blockCtx = ctx->blockStatement();
  if (blockCtx != nullptr) {
    // Create a new builder for this function body
    mlir::OpBuilder fnBuilder(context);
    fnBuilder.setInsertionPointToEnd(entry);

    // Visit statements inside block
    for (auto stmt : blockCtx->statement()) {
      // For now, just skip - not implementing full lowering yet
      (void)stmt;
    }

    // Ensure function has a return
    auto c0 = fnBuilder.create<mlir::arith::ConstantOp>(
        fnBuilder.getUnknownLoc(), fnBuilder.getIntegerAttr(i32Type, 0));
    fnBuilder.create<mlir::func::ReturnOp>(fnBuilder.getUnknownLoc(),
                                           c0.getResult());
  }

  // Add function to module
  module->push_back(func);
  return nullptr;
}

std::any
ASTBuilder::visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) {
  // Handle simple assignment - not implemented yet
  return nullptr;
}

std::any ASTBuilder::visitExpression(ZenithParser::ExpressionContext *ctx) {
  // Delegate to child visitor
  return visitChildren(ctx);
}

#else
// Non-MLIR stub implementations
ASTBuilder::ASTBuilder(mlir::MLIRContext *context) {}

std::any ASTBuilder::visitProgram(ZenithParser::ProgramContext *ctx) {
  return nullptr;
}

std::any ASTBuilder::visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) {
  return nullptr;
}

std::any
ASTBuilder::visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) {
  return nullptr;
}

std::any ASTBuilder::visitExpression(ZenithParser::ExpressionContext *ctx) {
  return nullptr;
}
#endif
