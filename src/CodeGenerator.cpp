#include "CodeGenerator.h"
#include <iostream>
#include <fstream>
#include <algorithm>

CodeGenerator::CodeGenerator() {}

void CodeGenerator::emitHeaders() {
  headers << "#include <stdio.h>\n";
  headers << "#include <stdlib.h>\n";
  headers << "#include <string.h>\n\n";
}

std::any CodeGenerator::visitProgram(ZenithParser::ProgramContext *ctx) {
  emitHeaders();

  // Visit all statements (generates function definitions, collect main info during traversal)
  for (auto stmt : ctx->statement()) {
    visit(stmt);
  }

  // After visiting, check if main was encountered. If not, add stub
  bool hasMain = std::find(functionNames.begin(), functionNames.end(), "main") != 
                 functionNames.end();

  if (!hasMain) {
    functions << "\nint main() {\n";
    functions << "  printf(\"No main function defined.\\n\");\n";
    functions << "  return 0;\n";
    functions << "}\n";
  }

  return nullptr;
}

std::any CodeGenerator::visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) {
  std::string funcName = ctx->IDENTIFIER()->getText();
  functionNames.push_back(funcName);  // Record that we've seen this function
  
  std::string returnType = "int";  // Default return type

  if (ctx->type()) {
    std::string typeStr = ctx->type()->getText();
    if (typeStr == "Int" || typeStr == "int") {
      returnType = "int";
    } else if (typeStr == "Float" || typeStr == "float") {
      returnType = "float";
    } else if (typeStr == "String" || typeStr == "string") {
      returnType = "const char *";
    }
  }

  functions << returnType << " " << funcName << "() {\n";

  // Generate function body
  if (ctx->blockStatement()) {
    visit(ctx->blockStatement());
  }

  functions << "  return 0;\n";
  functions << "}\n\n";

  return nullptr;
}

std::any CodeGenerator::visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) {
  std::string type = "int";  // Default type
  if (ctx->type()) {
    std::string typeStr = ctx->type()->getText();
    if (typeStr == "Float" || typeStr == "float") {
      type = "float";
    } else if (typeStr == "String" || typeStr == "string") {
      type = "const char *";
    }
  }

  if (ctx->identifierList()) {
    for (auto id : ctx->identifierList()->IDENTIFIER()) {
      std::string varName = id->getText();
      symbolTable[varName] = type;
      functions << "  " << type << " " << varName << " = 0;\n";
    }
  }

  return nullptr;
}

std::any CodeGenerator::visitExpression(ZenithParser::ExpressionContext *ctx) {
  if (!ctx) return nullptr;

  std::string result;
  // Simple expression handling - just get the text for now
  result = ctx->getText();

  return result;
}

std::any CodeGenerator::visitCallExpr(ZenithParser::CallExprContext *ctx) {
  if (!ctx) return nullptr;

  std::string primary = ctx->primaryExpr()->getText();
  auto suffixes = ctx->callSuffix();

  if (!suffixes.empty()) {
    // Function call
    std::string callStr = primary + "(";
    // TODO: Parse arguments from suffixes
    callStr += ")";
    functions << "  " << callStr << ";\n";
    return callStr;
  }

  return primary;
}

std::any CodeGenerator::visitPrimaryExpr(ZenithParser::PrimaryExprContext *ctx) {
  if (!ctx) return nullptr;
  return ctx->getText();
}

std::string CodeGenerator::getGeneratedCode() const {
  std::ostringstream result;
  result << headers.str();
  result << functions.str();
  return result.str();
}

void CodeGenerator::writeToFile(const std::string &filename) const {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  file << getGeneratedCode();
  file.close();
}
