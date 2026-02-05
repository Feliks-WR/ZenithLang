#include "ProofSolver.h"
#include "TypeChecker.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "ZenithParserBaseVisitor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

// ANTLR visitor
#include "ZenithParserBaseVisitor.h"

using namespace antlr4;
using namespace mlir::customlang;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: zenith <file.zenith> [--no-check-proofs]\n";
        std::cerr << "  --no-check-proofs: disable compile-time proof checking\n";
        return 1;
    }

    // Parse command line
    std::string inputFile = argv[1];
    bool checkProofs = true;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-check-proofs") {
            checkProofs = false;
        }
    }

    // Read input file
    std::ifstream file(inputFile);
    if (!file) {
        std::cerr << "Error: cannot open file '" << inputFile << "'\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    file.close();

    // Parse
    ANTLRInputStream input(source);
    ZenithLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    ZenithParser parser(&tokens);

    // Get AST
    ZenithParser::ProgramContext *tree = parser.program();

    if (parser.getNumberOfSyntaxErrors() > 0) {
        std::cerr << "❌ Parse errors detected\n";
        return 1;
    }

    std::cout << "✓ Parsed successfully\n";

    // Proof checking (if enabled)
    if (checkProofs) {
      TypeChecker checker;
      ProofSolver solver;

      // Walk AST: collect simple constant assignments and create obligations
      // Visitor that inspects multiplicative expressions and array indexing
      struct CheckerVisitor : public ZenithParserBaseVisitor {
        TypeChecker &checker;
        std::unordered_map<std::string, std::optional<long>> &constantValues;

        CheckerVisitor(TypeChecker &c,
                       std::unordered_map<std::string, std::optional<long>> &cv)
            : checker(c), constantValues(cv) {}

        antlrcpp::Any
        visitEquation(ZenithParser::EquationContext *ctx) override {
          // equation: expression EQUALS expression
          auto left = ctx->expression(0)->getText();
          auto right = ctx->expression(1)->getText();

          // If left is an identifier and right is integer literal, record
          // constant
          if (!left.empty() &&
              std::isalpha(static_cast<unsigned char>(left[0]))) {
            try {
              long v = std::stol(right);
              constantValues[left] = v;
              // register variable as int with single-value constraint
              checker.declareVariable(
                  left, mlir::customlang::DependentType::makeIntWithConstraint(
                            mlir::customlang::Constraint::makeSingleValue(v)));
              checker.assignVariable(left, right);
            } catch (...) {
              // not an integer literal
            }
          }

          return visitChildren(ctx);
        }

        antlrcpp::Any visitMultiplicativeExpr(
            ZenithParser::MultiplicativeExprContext *ctx) override {
          // Look for DIV or MOD operators in the text and create obligations
          std::string txt = ctx->getText();
          // Quick scan: find '/' or '%' occurrences
          for (size_t i = 0; i < txt.size(); ++i) {
            if (txt[i] == '/' || txt[i] == '%') {
              // Extract right-hand operand (divisor) by scanning rest of string
              size_t j = i + 1;
              while (j < txt.size() && txt[j] == ' ')
                ++j;
              size_t start = j;
              // read until next operator (+-*/% ) or end
              while (j < txt.size() && txt[j] != '/' && txt[j] != '%' &&
                     txt[j] != '+' && txt[j] != '-')
                ++j;
              std::string divisor = txt.substr(start, j - start);

              // determine location placeholder
              std::string loc = "(source)";

              auto dtype = checker.getVariableType(divisor);
              if (!dtype) {
                // create a generic int type for unknown
                checker.declareVariable(
                    divisor, mlir::customlang::DependentType::makeInt());
                dtype = checker.getVariableType(divisor);
              }

              if (txt[i] == '/') {
                checker.checkDivision(dtype, divisor, loc);
              } else {
                checker.checkModulo(dtype, divisor, loc);
              }
            }
          }

          return visitChildren(ctx);
        }

        antlrcpp::Any
        visitCallSuffix(ZenithParser::CallSuffixContext *ctx) override {
          // callSuffix could be array indexing: LBRACKET expression RBRACKET
          if (ctx->LBRACKET()) {
            // parent text contains array name and index; attempt to locate
            // index and array
            std::string txt = ctx->getText();
            // format is [index]; the parent callExpr will have the identifier
            // For simplicity, find the index expression inside brackets
            std::string inner = txt.substr(1, txt.size() - 2);
            // try to find array name via parent
            auto parent = ctx->parent;
            std::string arrName = "";
            if (parent) {
              arrName = parent->getText();
              // strip suffix from identifier if present
              size_t pos = arrName.find('[');
              if (pos != std::string::npos)
                arrName = arrName.substr(0, pos);
            }

            std::string loc = "(source)";

            auto arrType = checker.getVariableType(arrName);
            if (!arrType) {
              // assume array with unknown length; cannot prove bounds
              // still add an obligation using inner as subject and unknown
              // bound
              mlir::customlang::ProofObligation obl(
                  mlir::customlang::ProofObligation::ArrayBounds, loc,
                  "Index " + inner + " must be within bounds", inner, "?");
              checker.addObligation(obl);
            } else {
              checker.checkArrayAccess(arrType, inner, loc);
            }
          }

          return visitChildren(ctx);
        }
      };

      std::unordered_map<std::string, std::optional<long>> constantValues;
      CheckerVisitor visitor(checker, constantValues);

      // Walk the parse tree
      visitor.visitProgram(tree);

      std::cout << "✓ Proof checking enabled\n";

      if (checker.hasErrors()) {
        std::cerr << "❌ Type/proof errors:\n";
        for (const auto &err : checker.getErrors()) {
          std::cerr << "  " << err << "\n";
        }
        return 1;
      }

      if (!checker.getWarnings().empty()) {
        std::cout << "⚠ Warnings:\n";
        for (const auto &warn : checker.getWarnings()) {
          std::cout << "  " << warn << "\n";
        }
      }
    }

    std::cout << "✓ Compilation successful (AST mode)\n";
    std::cout << "  C transpiler removed - proof system active\n";

    return 0;
}
