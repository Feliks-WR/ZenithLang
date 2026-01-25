#ifndef CUSTOMLANG_ASTBUILDER_H
#define CUSTOMLANG_ASTBUILDER_H

#include "ZenithParserBaseVisitor.h"
#include <memory>
#include <string>
#include <unordered_map>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::customlang {

class ASTBuilder : public ZenithParserBaseVisitor {
 public:
  ASTBuilder(mlir::MLIRContext *context);

  // Visitors - to be implemented
  virtual std::any visitProgram(ZenithParser::ProgramContext *ctx) override;
  virtual std::any visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) override;
  virtual std::any visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) override;
  virtual std::any visitExpression(ZenithParser::ExpressionContext *ctx) override;

  #ifdef USE_MLIR
  mlir::OwningOpRef<mlir::ModuleOp> getModule() { return std::move(module); }
  #else
  void *getModule() { return nullptr; }
  #endif

 private:
  mlir::MLIRContext *context;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unordered_map<std::string, mlir::Value> symbolTable;
};

}  // namespace mlir::customlang

#endif  // CUSTOMLANG_ASTBUILDER_H
