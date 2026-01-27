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
    } else if (typeStr == "Void" || typeStr == "void") {
      returnType = "void";
    }
  }

  // Handle parameters
  std::string params = "";
  if (ctx->parameterList()) {
    std::vector<std::string> paramStrs;
    for (auto param : ctx->parameterList()->parameter()) {
      std::string paramType = "int";  // Default
      if (param->type()) {
        std::string typeStr = param->type()->getText();
        if (typeStr == "float" || typeStr == "Float") paramType = "float";
        else if (typeStr == "string" || typeStr == "String") paramType = "const char*";
      }
      std::string paramName = param->IDENTIFIER()->getText();
      paramStrs.push_back(paramType + " " + paramName);
    }
    params = "";
    for (size_t i = 0; i < paramStrs.size(); i++) {
      params += paramStrs[i];
      if (i < paramStrs.size() - 1) params += ", ";
    }
  }

  functions << returnType << " " << funcName << "(" << params << ") {\n";

  // Clear symbol table for new function scope
  symbolTable.clear();

  // Generate function body
  if (ctx->blockStatement()) {
    visit(ctx->blockStatement());
  } else if (ctx->expression()) {
    // Function with expression body: f() = expr
    auto exprResult = visit(ctx->expression());
    if (exprResult.has_value()) {
      std::string exprStr = std::any_cast<std::string>(exprResult);
      functions << "  return " << exprStr << ";\n";
    }
  }

  // Only add default return for non-void functions
  if (returnType != "void") {
    functions << "  return 0;\n";
  }
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

std::any CodeGenerator::visitReturnStatement(ZenithParser::ReturnStatementContext *ctx) {
  if (!ctx) return nullptr;
  
  functions << "  return";
  if (ctx->expression()) {
    auto exprResult = visit(ctx->expression());
    if (exprResult.has_value()) {
      std::string exprStr = std::any_cast<std::string>(exprResult);
      functions << " " << exprStr;
    }
  }
  functions << ";\n";
  return nullptr;
}

std::any CodeGenerator::visitPrintStatement(ZenithParser::PrintStatementContext *ctx) {
  if (!ctx) return nullptr;
  
  // Generate printf for each expression
  for (auto expr : ctx->expression()) {
    auto exprResult = visit(expr);
    if (exprResult.has_value()) {
      std::string exprStr = std::any_cast<std::string>(exprResult);
      
      // Determine format specifier based on expression type
      // Simple heuristic: if it starts with ", it's a string
      if (exprStr[0] == '"') {
        functions << "  printf(\"%s\\n\", " << exprStr << ");\n";
      } else if (exprStr.find('.') != std::string::npos) {
        functions << "  printf(\"%f\\n\", " << exprStr << ");\n";
      } else {
        functions << "  printf(\"%d\\n\", " << exprStr << ");\n";
      }
    }
  }
  return nullptr;
}

std::any CodeGenerator::visitIfStatement(ZenithParser::IfStatementContext *ctx) {
  if (!ctx) return nullptr;
  
  auto condResult = visit(ctx->expression());
  if (condResult.has_value()) {
    std::string condStr = std::any_cast<std::string>(condResult);
    functions << "  if (" << condStr << ") {\n";
  }
  
  // Visit if body
  if (ctx->blockStatement().size() > 0) {
    visit(ctx->blockStatement(0));
  }
  
  functions << "  }\n";
  
  // Handle else clause
  if (ctx->ELSE()) {
    functions << "  else {\n";
    if (ctx->blockStatement().size() > 1) {
      visit(ctx->blockStatement(1));
    }
    functions << "  }\n";
  }
  
  return nullptr;
}

std::any CodeGenerator::visitWhileStatement(ZenithParser::WhileStatementContext *ctx) {
  if (!ctx) return nullptr;
  
  auto condResult = visit(ctx->expression());
  if (condResult.has_value()) {
    std::string condStr = std::any_cast<std::string>(condResult);
    functions << "  while (" << condStr << ") {\n";
  }
  
  if (ctx->blockStatement()) {
    visit(ctx->blockStatement());
  }
  
  functions << "  }\n";
  return nullptr;
}

std::any CodeGenerator::visitEquation(ZenithParser::EquationContext *ctx) {
  if (!ctx) return nullptr;
  
  // Handle variable assignment: x = expr
  auto exprs = ctx->expression();
  if (exprs.size() == 2) {
    std::string lhs = std::any_cast<std::string>(visit(exprs[0]));
    std::string rhs = std::any_cast<std::string>(visit(exprs[1]));
    
    // Always declare variables when assigning (simple approach)
    // Check if it looks like a simple identifier (no operators/function calls)
    bool isSimpleIdentifier = (lhs.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string::npos);
    
    if (isSimpleIdentifier && symbolTable.find(lhs) == symbolTable.end()) {
      // Declare it as int by default
      symbolTable[lhs] = "int";
      functions << "  int " << lhs << " = " << rhs << ";\n";
    } else {
      functions << "  " << lhs << " = " << rhs << ";\n";
    }
  }
  
  return nullptr;
}

std::any CodeGenerator::visitBlockStatement(ZenithParser::BlockStatementContext *ctx) {
  if (!ctx) return nullptr;
  
  for (auto stmt : ctx->statement()) {
    visit(stmt);
  }
  
  return nullptr;
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
