#ifndef CUSTOMLANG_ASTBUILDER_H
#define CUSTOMLANG_ASTBUILDER_H

#include "CustomLangBaseVisitor.h"
#include "mlir/IR/Builder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ModuleOp.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace mlir::customlang {

class ASTBuilder : public CustomLangBaseVisitor {
 public:
  ASTBuilder(mlir::MLIRContext *context);

  // Visitors - to be implemented
  virtual std::any visitProgram(CustomLangParser::ProgramContext *ctx) override;
  virtual std::any visitFunctionDecl(CustomLangParser::FunctionDeclContext *ctx) override;
  virtual std::any visitVarDecl(CustomLangParser::VarDeclContext *ctx) override;
  virtual std::any visitExpression(CustomLangParser::ExpressionContext *ctx) override;

  mlir::OwningOpRef<mlir::ModuleOp> getModule() { return std::move(module); }

 private:
  mlir::MLIRContext *context;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unordered_map<std::string, mlir::Value> symbolTable;
};

}  // namespace mlir::customlang

#endif  // CUSTOMLANG_ASTBUILDER_H
