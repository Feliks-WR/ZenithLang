#include "CodeGenerator.h"
#include <algorithm>
#include <fstream>
#include <iostream>

CodeGenerator::CodeGenerator() {}

void CodeGenerator::emitHeaders() {
  headers << "#include <stdio.h>\n";
  headers << "#include <stdlib.h>\n";
  headers << "#include <string.h>\n\n";
}

std::any CodeGenerator::visitProgram(ZenithParser::ProgramContext *ctx) {
  emitHeaders();

  // Visit all statements (generates function definitions, collect main info
  // during traversal)
  for (auto stmt : ctx->statement()) {
    visit(stmt);
  }

  // After visiting, check if main was encountered. If not, add stub
  bool hasMain = std::find(functionNames.begin(), functionNames.end(),
                           "main") != functionNames.end();

  if (!hasMain) {
    functions << "\nint main() {\n";
    functions << "  printf(\"No main function defined.\\n\");\n";
    functions << "  return 0;\n";
    functions << "}\n";
  }

  return nullptr;
}

std::any
CodeGenerator::visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) {
  std::string funcName = ctx->IDENTIFIER()->getText();
  functionNames.push_back(funcName); // Record that we've seen this function

  std::string returnType = "int"; // Default return type

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
      std::string paramType = "int"; // Default
      if (param->type()) {
        std::string typeStr = param->type()->getText();
        if (typeStr == "float" || typeStr == "Float")
          paramType = "float";
        else if (typeStr == "string" || typeStr == "String")
          paramType = "const char*";
      }
      std::string paramName = param->IDENTIFIER()->getText();
      paramStrs.push_back(paramType + " " + paramName);
    }
    params = "";
    for (size_t i = 0; i < paramStrs.size(); i++) {
      params += paramStrs[i];
      if (i < paramStrs.size() - 1)
        params += ", ";
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

std::any
CodeGenerator::visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) {
  std::string type = "int"; // Default type
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
  if (!ctx)
    return nullptr;

  std::string result;
  // Simple expression handling - just get the text for now
  result = ctx->getText();

  return result;
}

std::any CodeGenerator::visitCallExpr(ZenithParser::CallExprContext *ctx) {
  if (!ctx)
    return nullptr;

  std::string primary = std::any_cast<std::string>(visit(ctx->primaryExpr()));
  auto suffixes = ctx->callSuffix();

  if (!suffixes.empty()) {
    for (auto suffix : suffixes) {
      if (suffix->LBRACKET()) {
        // Array indexing: arr[i] or slicing: arr[i..j]
        auto sliceOrIdx = suffix->sliceOrIndex();
        if (sliceOrIdx) {
          auto exprs = sliceOrIdx->expression();
          if (exprs.size() == 1 && !sliceOrIdx->DOTDOT()) {
            // Simple indexing
            std::string index = std::any_cast<std::string>(visit(exprs[0]));
            primary = primary + "[" + index + "]";
          } else {
            // Slicing - complex, not fully implemented yet
            std::string sliceSpec =
                std::any_cast<std::string>(visit(sliceOrIdx));
            primary = "/* slice: " + primary + "[" + sliceSpec + "] */";
          }
        }
      } else if (suffix->DOT()) {
        // Member access: obj.field or arr.length
        std::string member = suffix->IDENTIFIER()->getText();
        if (member == "length") {
          // Array length access
          if (arrayLengths.find(primary) != arrayLengths.end()) {
            primary = std::to_string(arrayLengths[primary]);
          } else {
            primary = "sizeof(" + primary + ")/sizeof(" + primary + "[0])";
          }
        } else {
          primary = primary + "." + member;
        }
      } else if (suffix->LPAREN()) {
        // Function call: f(args)
        std::string callStr = primary + "(";
        auto args = suffix->expression();
        for (size_t i = 0; i < args.size(); i++) {
          auto argResult = visit(args[i]);
          if (argResult.has_value()) {
            callStr += std::any_cast<std::string>(argResult);
            if (i < args.size() - 1)
              callStr += ", ";
          }
        }
        callStr += ")";
        primary = callStr;
      }
    }
  }

  return primary;
}

std::any
CodeGenerator::visitPrimaryExpr(ZenithParser::PrimaryExprContext *ctx) {
  if (!ctx)
    return nullptr;

  // Check for array literal first
  if (ctx->arrayLiteral()) {
    return visit(ctx->arrayLiteral());
  }

  return ctx->getText();
}

std::any
CodeGenerator::visitReturnStatement(ZenithParser::ReturnStatementContext *ctx) {
  if (!ctx)
    return nullptr;

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

std::any
CodeGenerator::visitPrintStatement(ZenithParser::PrintStatementContext *ctx) {
  if (!ctx)
    return nullptr;

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

std::any
CodeGenerator::visitIfStatement(ZenithParser::IfStatementContext *ctx) {
  if (!ctx)
    return nullptr;

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

std::any
CodeGenerator::visitWhileStatement(ZenithParser::WhileStatementContext *ctx) {
  if (!ctx)
    return nullptr;

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
  if (!ctx)
    return nullptr;

  // Handle variable assignment with =: x = expr
  auto exprs = ctx->expression();
  if (exprs.size() == 2) {
    std::string lhs = std::any_cast<std::string>(visit(exprs[0]));
    std::string rhs = std::any_cast<std::string>(visit(exprs[1]));

    // Always declare variables when assigning (simple approach)
    // Check if it looks like a simple identifier (no operators/function calls)
    bool isSimpleIdentifier =
        (lhs.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU"
                               "VWXYZ0123456789_") == std::string::npos);

    if (isSimpleIdentifier && symbolTable.find(lhs) == symbolTable.end()) {
      // Declare it as int by default (deeply immutable with =)
      symbolTable[lhs] = "int";
      functions << "  const int " << lhs << " = " << rhs << ";\n";
    } else {
      functions << "  " << lhs << " = " << rhs << ";\n";
    }
  }

  return nullptr;
}

std::any CodeGenerator::visitAssignment(ZenithParser::AssignmentContext *ctx) {
  if (!ctx)
    return nullptr;

  // Handle variable assignment with :=: x := expr
  // := means shallow const - can't reassign variable, but elements are mutable
  auto exprs = ctx->expression();
  if (exprs.size() == 2) {
    std::string lhs = std::any_cast<std::string>(visit(exprs[0]));
    std::string rhs = std::any_cast<std::string>(visit(exprs[1]));

    bool isSimpleIdentifier =
        (lhs.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU"
                               "VWXYZ0123456789_") == std::string::npos);

    if (isSimpleIdentifier && symbolTable.find(lhs) == symbolTable.end()) {
      symbolTable[lhs] = "int"; // Will be updated based on RHS
      isShallowConst[lhs] = true;

      // Check if RHS is an array literal
      if (rhs.find("__arr_") != std::string::npos) {
        // Array assignment - extract array info
        functions << "  int * const " << lhs << " = " << rhs << ";\n";
      } else {
        functions << "  int const " << lhs << " = " << rhs << ";\n";
      }
    } else {
      functions << "  " << lhs << " = " << rhs << ";\n";
    }
  }

  return nullptr;
}

std::any
CodeGenerator::visitArrayLiteral(ZenithParser::ArrayLiteralContext *ctx) {
  if (!ctx)
    return nullptr;

  // Generate: { expr1, expr2, ... }
  std::ostringstream arrCode;
  std::vector<std::string> elements;

  for (auto expr : ctx->expression()) {
    auto result = visit(expr);
    if (result.has_value()) {
      elements.push_back(std::any_cast<std::string>(result));
    }
  }

  // Generate a static array in C
  std::string tempName = "__arr_" + std::to_string(tempVarCounter++);
  int length = elements.size();

  functions << "  static int " << tempName << "[" << length << "] = {";
  for (size_t i = 0; i < elements.size(); i++) {
    functions << elements[i];
    if (i < elements.size() - 1)
      functions << ", ";
  }
  functions << "};\n";

  // Store length for bounds checking
  arrayLengths[tempName] = length;

  return tempName;
}

std::any
CodeGenerator::visitSliceOrIndex(ZenithParser::SliceOrIndexContext *ctx) {
  if (!ctx)
    return nullptr;

  auto exprs = ctx->expression();
  if (exprs.size() == 1) {
    // Simple indexing: arr[i]
    return std::any_cast<std::string>(visit(exprs[0]));
  } else if (exprs.size() == 2) {
    // Slicing: arr[start..end]
    std::string start = std::any_cast<std::string>(visit(exprs[0]));
    std::string end = std::any_cast<std::string>(visit(exprs[1]));

    // For slicing, we'll need to generate a new array
    // This is complex - for now, just mark it
    return start + ".." + end; // Placeholder
  } else if (ctx->DOTDOT()) {
    // Open-ended slice: arr[start..]
    std::string start = std::any_cast<std::string>(visit(exprs[0]));
    return start + ".."; // Placeholder
  }

  return nullptr;
}

std::any
CodeGenerator::visitBlockStatement(ZenithParser::BlockStatementContext *ctx) {
  if (!ctx)
    return nullptr;

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
