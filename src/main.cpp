#include "ProofSolver.h"
#include "TypeChecker.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "ZenithParserBaseVisitor.h"

#ifdef USE_MLIR
#include "ASTBuilder.h"
#include "Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#endif

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

struct Options {
  std::string input_path;
  std::string output_path;
  bool check_proofs = true;
  bool dump_ir = false;
};

void PrintUsage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " <file.zenith> [-o <file.ll>] [--no-check-proofs] [--dump-ir]\n";
  std::cerr << "  -o <file.ll>: write LLVM IR to this file\n";
  std::cerr << "  --no-check-proofs: disable compile-time proof checking\n";
  std::cerr << "  --dump-ir: print LLVM IR to stdout before execution\n";
}

bool ReadFileToString(const std::string &path, std::string *out,
                      std::string *error) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    if (error) {
      *error = "error: cannot open file '" + path + "'";
    }
    return false;
  }

  std::ostringstream buffer;
  buffer << file.rdbuf();
  *out = buffer.str();
  return true;
}

std::string DefaultOutputPath(const std::string &input_path) {
  std::string output = input_path;
  std::string::size_type last_slash = output.find_last_of("/\\");
  std::string::size_type last_dot = output.find_last_of('.');

  if (last_dot != std::string::npos &&
      (last_slash == std::string::npos || last_dot > last_slash)) {
    output.replace(last_dot, std::string::npos, ".mlir");
  } else {
    output += ".mlir";
  }

  return output;
}

bool ParseArgs(int argc, char **argv, Options *options, std::string *error) {
  if (argc < 2) {
    if (error) {
      *error = "error: missing input file";
    }
    return false;
  }

  options->input_path = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--no-check-proofs") {
      options->check_proofs = false;
    } else if (arg == "--dump-mlir" || arg == "--dump-ir") {
      options->dump_ir = true;
    } else if (arg == "-o") {
      if (i + 1 >= argc) {
        if (error) {
          *error = "error: missing argument for -o";
        }
        return false;
      }
      options->output_path = argv[++i];
    } else {
      if (error) {
        *error = "error: unknown argument '" + arg + "'";
      }
      return false;
    }
  }

  if (options->output_path.empty()) {
    options->output_path = DefaultOutputPath(options->input_path);
  }

  return true;
}

bool IsIdentifier(const std::string &text) {
  if (text.empty()) {
    return false;
  }

  unsigned char first = static_cast<unsigned char>(text[0]);
  if (!std::isalpha(first) && text[0] != '_') {
    return false;
  }

  for (size_t i = 1; i < text.size(); ++i) {
    unsigned char ch = static_cast<unsigned char>(text[i]);
    if (!std::isalnum(ch) && text[i] != '_') {
      return false;
    }
  }

  return true;
}

std::optional<long> ParseIntegerLiteral(const std::string &text) {
  if (text.empty()) {
    return std::nullopt;
  }

  char *end = nullptr;
  errno = 0;
  long value = std::strtol(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') {
    return std::nullopt;
  }
  return value;
}

class CheckerVisitor : public ZenithParserBaseVisitor {
public:
  CheckerVisitor(
      mlir::customlang::TypeChecker &checker,
      std::unordered_map<std::string, std::optional<long>> &constant_values)
      : checker_(checker), constant_values_(constant_values) {}

  antlrcpp::Any visitEquation(ZenithParser::EquationContext *ctx) override {
    std::string left = ctx->expression(0)->getText();
    std::string right = ctx->expression(1)->getText();

    if (IsIdentifier(left)) {
      auto value = ParseIntegerLiteral(right);
      if (value.has_value()) {
        constant_values_[left] = value.value();
        checker_.declareVariable(
            left,
            mlir::customlang::DependentType::makeIntWithConstraint(
                mlir::customlang::Constraint::makeSingleValue(value.value())));
        checker_.assignVariable(left, right);
      }
    }

    return visitChildren(ctx);
  }

  antlrcpp::Any visitMultiplicativeExpr(
      ZenithParser::MultiplicativeExprContext *ctx) override {
    std::string text = ctx->getText();
    for (size_t i = 0; i < text.size(); ++i) {
      if (text[i] != '/' && text[i] != '%') {
        continue;
      }

      size_t start = i + 1;
      size_t end = start;
      while (end < text.size() && text[end] != '/' && text[end] != '%' &&
             text[end] != '+' && text[end] != '-') {
        ++end;
      }

      std::string divisor = text.substr(start, end - start);
      if (divisor.empty()) {
        continue;
      }

      std::string location = "(source)";
      auto dtype = checker_.getVariableType(divisor);
      if (!dtype) {
        auto literal = ParseIntegerLiteral(divisor);
        if (literal.has_value()) {
          dtype = mlir::customlang::DependentType::makeIntWithConstraint(
              mlir::customlang::Constraint::makeSingleValue(literal.value()));
        } else {
          checker_.declareVariable(divisor,
                                   mlir::customlang::DependentType::makeInt());
          dtype = checker_.getVariableType(divisor);
        }
      }

      if (text[i] == '/') {
        checker_.checkDivision(dtype, divisor, location);
      } else {
        checker_.checkModulo(dtype, divisor, location);
      }
    }

    return visitChildren(ctx);
  }

  antlrcpp::Any visitCallSuffix(ZenithParser::CallSuffixContext *ctx) override {
    if (!ctx->LBRACKET()) {
      return visitChildren(ctx);
    }

    std::string suffix_text = ctx->getText();
    std::string index_expr = suffix_text.substr(1, suffix_text.size() - 2);
    std::string array_name;

    if (auto *parent = ctx->parent) {
      array_name = parent->getText();
      size_t pos = array_name.find('[');
      if (pos != std::string::npos) {
        array_name = array_name.substr(0, pos);
      }
    }

    std::string location = "(source)";
    auto array_type = checker_.getVariableType(array_name);
    if (!array_type) {
      mlir::customlang::ProofObligation obligation(
          mlir::customlang::ProofObligation::ArrayBounds, location,
          "Index " + index_expr + " must be within bounds", index_expr, "?");
      checker_.addObligation(obligation);
    } else {
      checker_.checkArrayAccess(array_type, index_expr, location);
    }

    return visitChildren(ctx);
  }

private:
  mlir::customlang::TypeChecker &checker_;
  std::unordered_map<std::string, std::optional<long>> &constant_values_;
};

} // namespace

int main(int argc, char **argv) {
  Options options;
  std::string error;
  if (!ParseArgs(argc, argv, &options, &error)) {
    if (!error.empty()) {
      std::cerr << error << "\n";
    }
    PrintUsage(argv[0]);
    return 1;
  }

  std::string source;
  if (!ReadFileToString(options.input_path, &source, &error)) {
    std::cerr << error << "\n";
    return 1;
  }

  antlr4::ANTLRInputStream input(source);
  ZenithLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  ZenithParser parser(&tokens);

  auto *tree = parser.program();
  if (parser.getNumberOfSyntaxErrors() > 0) {
    std::cerr << "parse error: syntax errors detected\n";
    return 1;
  }

  std::cout << "✓ parsed successfully\n";

  if (options.check_proofs) {
    mlir::customlang::TypeChecker checker;
    mlir::customlang::ProofSolver solver;

    std::unordered_map<std::string, std::optional<long>> constant_values;
    CheckerVisitor visitor(checker, constant_values);
    visitor.visitProgram(tree);

    checker.requireProofs(solver, constant_values);

    if (checker.hasErrors()) {
      std::cerr << "error: type/proof errors:\n";
      for (const auto &err : checker.getErrors()) {
        std::cerr << "  " << err << "\n";
      }
      return 1;
    }

    if (!checker.getWarnings().empty()) {
      std::cout << "⚠ warnings:\n";
      for (const auto &warn : checker.getWarnings()) {
        std::cout << "  " << warn << "\n";
      }
    }
  }

  std::cout << "✓ proof checking complete\n";

  // Initialize MLIR context and generate MLIR.
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::customlang::CustomLangDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();

  mlir::customlang::ASTBuilder builder(&context);
  builder.visitProgram(tree);

  auto module = builder.getModule();
  if (!module) {
    std::cerr << "error: failed to generate MLIR module\n";
    return 1;
  }

  if (options.dump_ir) {
    std::cout << "\n--- MLIR ---\n";
    module->print(llvm::outs());
    std::cout << "\n";
  }

  // Write MLIR to file if requested
  std::error_code ec;
  llvm::raw_fd_ostream out_file(options.output_path, ec);
  if (ec) {
    std::cerr << "error: cannot open " << options.output_path
              << " for writing\n";
    return 1;
  }
  module->print(out_file);
  out_file.close();

  std::cout << "✓ MLIR generated and written to " << options.output_path
            << "\n";
  std::cout << "✓ compilation successful (MLIR dialect mode)\n";

  return 0;
}
