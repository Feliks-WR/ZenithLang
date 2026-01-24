#include "ASTBuilder.h"
#ifdef USE_MLIR
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#endif

using namespace mlir::customlang;

ASTBuilder::ASTBuilder(mlir::MLIRContext *context)
#ifdef USE_MLIR
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
  auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(), name, funcType);
  mlir::Block *entry = func.addEntryBlock();
  mlir::OpBuilder fnBuilder(entry);

  // Clear symbol table for new function
  symbolTable.clear();

  // Lower function body if present (blockStatement)
  auto blocks = ctx->blockStatement();
  if (!blocks.empty()) {
    auto block = blocks[0];
    // Visit statements inside block
    // Set insertion point to end of entry
    fnBuilder.setInsertionPointToEnd(entry);
    // Temporarily swap builders
    auto prevBuilder = this->builder;
    this->builder = fnBuilder;

    for (auto stmt : block->statement()) {
      visit(stmt);
    }

    // Ensure function has a return
    if (entry->empty() || !llvm::isa<mlir::func::ReturnOp>(entry->getTerminator())) {
      // default return 0
      auto c0 = this->builder.create<mlir::arith::ConstantOp>(this->builder.getUnknownLoc(), this->builder.getIntegerAttr(i32Type, 0));
      this->builder.create<mlir::func::ReturnOp>(this->builder.getUnknownLoc(), c0.getResult());
    }

    this->builder = prevBuilder;
  }

  module.push_back(func);
  return nullptr;
}

std::any ASTBuilder::visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) {
  // Handle simple assignment in form: identifier = expression (via equation) or varDeclaration without init ignored
  return nullptr;
}

// Expressions: basic integer constants, identifiers, and binary plus
std::any ASTBuilder::visitExpression(ZenithParser::ExpressionContext *ctx) {
  // Delegate to child visitor; return the mlir::Value via std::any
  return visitChildren(ctx);
}

std::any ASTBuilder::visitAdditiveExpr(ZenithParser::AdditiveExprContext *ctx) {
  if (ctx->multiplicativeExpr().size() == 1 && ctx->PLUS().empty()) {
    return visit(ctx->multiplicativeExpr(0));
  }
  // fold left-associative
  mlir::Value acc;
  for (size_t i = 0; i < ctx->multiplicativeExpr().size(); ++i) {
    auto child = ctx->multiplicativeExpr(i);
    std::any anyv = visit(child);
    mlir::Value v = anyv.has_value() ? std::any_cast<mlir::Value>(anyv) : mlir::Value();
    if (i == 0) acc = v;
    else {
      acc = this->builder.create<mlir::arith::AddIOp>(this->builder.getUnknownLoc(), acc, v);
    }
  }
  return acc;
}

std::any ASTBuilder::visitPrimaryExpr(ZenithParser::PrimaryExprContext *ctx) {
  if (ctx->INTEGER()) {
    int val = std::stoi(ctx->INTEGER()->getText());
    auto i32Type = this->builder.getI32Type();
    auto c = this->builder.create<mlir::arith::ConstantOp>(this->builder.getUnknownLoc(), this->builder.getIntegerAttr(i32Type, val));
    return c.getResult();
  }
  if (ctx->IDENTIFIER()) {
    std::string name = ctx->IDENTIFIER()->getText();
    auto it = symbolTable.find(name);
    if (it != symbolTable.end()) return it->second;
    return mlir::Value();
  }
  return mlir::Value();
}

#else
ASTBuilder::ASTBuilder(void *context) {}

std::any ASTBuilder::visitProgram(ZenithParser::ProgramContext *ctx) { return nullptr; }
std::any ASTBuilder::visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) { return nullptr; }
std::any ASTBuilder::visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) { return nullptr; }
std::any ASTBuilder::visitExpression(ZenithParser::ExpressionContext *ctx) { return nullptr; }
#endif
