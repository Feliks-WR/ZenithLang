#ifndef ZENITH_CODEGENERATOR_H
#define ZENITH_CODEGENERATOR_H

#include "ZenithParserBaseVisitor.h"
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

class CodeGenerator : public ZenithParserBaseVisitor {
 public:
  CodeGenerator();

  // Visitor methods for code generation
  std::any visitProgram(ZenithParser::ProgramContext *ctx) override;
  std::any visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) override;
  std::any visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) override;
  std::any visitExpression(ZenithParser::ExpressionContext *ctx) override;
  std::any visitReturnStatement(ZenithParser::ReturnStatementContext *ctx) override;
  std::any visitPrintStatement(ZenithParser::PrintStatementContext *ctx) override;
  std::any visitIfStatement(ZenithParser::IfStatementContext *ctx) override;
  std::any visitWhileStatement(ZenithParser::WhileStatementContext *ctx) override;
  std::any visitEquation(ZenithParser::EquationContext *ctx) override;
  std::any visitBlockStatement(ZenithParser::BlockStatementContext *ctx) override;
  std::any visitCallExpr(ZenithParser::CallExprContext *ctx) override;
  std::any visitPrimaryExpr(ZenithParser::PrimaryExprContext *ctx) override;

  // Get generated C code
  std::string getGeneratedCode() const;

  // Helper to write to a file
  void writeToFile(const std::string &filename) const;

 private:
  std::ostringstream code;
  std::ostringstream headers;
  std::ostringstream functions;
  std::unordered_map<std::string, std::string> symbolTable;  // var_name -> type
  std::vector<std::string> functionNames;
  
  void emitHeaders();
  void emitMain();
};

#endif // ZENITH_CODEGENERATOR_H
