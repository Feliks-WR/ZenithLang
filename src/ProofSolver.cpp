#include "ProofSolver.h"
#include "TypeChecker.h"
#include <algorithm>
#include <cctype>
#include <sstream>

#ifdef USE_Z3
#include <z3++.h>
#endif

using namespace mlir::customlang;

ProofSolver::ProofSolver() = default;

ProofResult
ProofSolver::prove(const ProofObligation &obligation, const TypeEnv &typeEnv,
                   const std::unordered_map<std::string, std::optional<long>>
                       &constantValues) const {

  // Try constant-based proof first
  auto constResult = proveWithConstants(obligation, constantValues);
  if (constResult.proved) {
    return constResult;
  }

  // Try type-based proof
  auto typeResult = proveWithTypeConstraints(obligation, typeEnv);
  if (typeResult.proved) {
    return typeResult;
  }

#ifdef USE_Z3
  // Try Z3 SMT solver
  auto z3Result = proveWithZ3(obligation, typeEnv, constantValues);
  if (z3Result.proved) {
    return z3Result;
  }
#endif

  // Failed to prove
  return ProofResult(false, "Unable to discharge proof obligation");
}

ProofResult ProofSolver::proveWithConstants(
    const ProofObligation &obligation,
    const std::unordered_map<std::string, std::optional<long>> &constantValues)
    const {

  switch (obligation.kind) {
  case ProofObligation::DivisionNonZero:
  case ProofObligation::ModuloNonZero: {
    auto val = tryEvaluateConstant(obligation.subjectExpr, constantValues);
    if (val.has_value() && val.value() != 0) {
      return ProofResult(true, "Divisor is constant " +
                                   std::to_string(val.value()) + " != 0");
    }
    break;
  }

  case ProofObligation::ArrayBounds: {
    auto indexVal = tryEvaluateConstant(obligation.subjectExpr, constantValues);
    auto boundVal = tryEvaluateConstant(obligation.boundExpr, constantValues);

    if (indexVal.has_value() && boundVal.has_value()) {
      if (indexVal.value() >= 0 && indexVal.value() < boundVal.value()) {
        return ProofResult(true, "Index " + std::to_string(indexVal.value()) +
                                     " < bound " +
                                     std::to_string(boundVal.value()));
      }
    }
    break;
  }

  default:
    break;
  }

  return ProofResult(false, "Cannot prove with known constants");
}

ProofResult
ProofSolver::proveWithTypeConstraints(const ProofObligation &obligation,
                                      const TypeEnv &typeEnv) const {

  switch (obligation.kind) {
  case ProofObligation::DivisionNonZero:
  case ProofObligation::ModuloNonZero: {
    if (checkNonZeroFromType(obligation.subjectExpr, typeEnv)) {
      return ProofResult(true, "Divisor type guarantees non-zero");
    }
    break;
  }

  case ProofObligation::ArrayBounds: {
    if (checkBoundsFromType(obligation.subjectExpr, obligation.boundExpr,
                            typeEnv)) {
      return ProofResult(true, "Index type guarantees valid bounds");
    }
    break;
  }

  default:
    break;
  }

  return ProofResult(false, "Cannot prove from type constraints");
}

std::optional<long> ProofSolver::tryEvaluateConstant(
    const std::string &expr,
    const std::unordered_map<std::string, std::optional<long>> &constantValues)
    const {

  // Try direct lookup
  auto it = constantValues.find(expr);
  if (it != constantValues.end()) {
    return it->second;
  }

  // Try parsing as integer literal
  if (!expr.empty() && (std::isdigit(expr[0]) || expr[0] == '-')) {
    try {
      return std::stol(expr);
    } catch (...) {
      // Not a valid integer
    }
  }

  return std::nullopt;
}

bool ProofSolver::checkNonZeroFromType(const std::string &expr,
                                       const TypeEnv &typeEnv) const {
  auto type = typeEnv.getType(expr);
  if (!type) {
    return false;
  }

  // Check constraints for non-zero guarantees
  for (const auto &constraint : type->constraints) {
    if (constraint->kind == Constraint::Predicate) {
      const std::string &pred = constraint->expression;
      if (pred.find("!= 0") != std::string::npos ||
          pred.find("(!=0)") != std::string::npos) {
        return true;
      }
      if (pred.find("> 0") != std::string::npos ||
          pred.find(">= 1") != std::string::npos ||
          pred.find("< 0") != std::string::npos ||
          pred.find("<= -1") != std::string::npos) {
        return true;
      }
    }

    if (constraint->kind == Constraint::Range) {
      if (constraint->minValue > 0 || constraint->maxValue < 0) {
        return true;
      }
    }
  }

  return false;
}

bool ProofSolver::checkBoundsFromType(const std::string &indexExpr,
                                      const std::string &boundExpr,
                                      const TypeEnv &typeEnv) const {
  auto indexType = typeEnv.getType(indexExpr);
  if (!indexType) {
    return false;
  }

  // Check if index has range constraint that fits within bound
  for (const auto &constraint : indexType->constraints) {
    if (constraint->kind == Constraint::Range) {
      // If we can parse bound as integer and max < bound, we're safe
      try {
        long bound = std::stol(boundExpr);
        if (constraint->minValue >= 0 && constraint->maxValue < bound) {
          return true;
        }
      } catch (...) {
        // Bound is not a constant, cannot prove
      }
    }
  }

  return false;
}

#ifdef USE_Z3
ProofResult ProofSolver::proveWithZ3(
    const ProofObligation &obligation, const TypeEnv &typeEnv,
    const std::unordered_map<std::string, std::optional<long>> &constantValues)
    const {

  try {
    z3::context ctx;
    z3::solver solver(ctx);

    // Create Z3 variables for all symbols
    std::unordered_map<std::string, z3::expr> z3Vars;

    // Helper to get or create Z3 variable
    auto getZ3Var = [&](const std::string &name) -> z3::expr {
      auto it = z3Vars.find(name);
      if (it != z3Vars.end()) {
        return it->second;
      }

      // Check if this is a constant value
      auto constVal = tryEvaluateConstant(name, constantValues);
      if (constVal.has_value()) {
        return ctx.int_val(static_cast<int>(constVal.value()));
      }

      // Create new integer variable
      auto [insertedIt, inserted] =
          z3Vars.emplace(name, ctx.int_const(name.c_str()));
      z3::expr &var = insertedIt->second;

      // Add type constraints if available
      auto type = typeEnv.getType(name);
      if (type) {
        for (const auto &constraint : type->constraints) {
          if (constraint->kind == Constraint::Range) {
            solver.add(var >= static_cast<int>(constraint->minValue));
            solver.add(var <= static_cast<int>(constraint->maxValue));
          } else if (constraint->kind == Constraint::Predicate) {
            // Parse simple predicates
            const std::string &pred = constraint->expression;
            if (pred.find("!= 0") != std::string::npos) {
              solver.add(var != 0);
            } else if (pred.find("> 0") != std::string::npos) {
              solver.add(var > 0);
            } else if (pred.find(">= 0") != std::string::npos) {
              solver.add(var >= 0);
            } else if (pred.find("< 0") != std::string::npos) {
              solver.add(var < 0);
            }
          }
        }
      }

      return var;
    };

    // Build the proof obligation as a Z3 constraint
    switch (obligation.kind) {
    case ProofObligation::DivisionNonZero:
    case ProofObligation::ModuloNonZero: {
      z3::expr divisor = getZ3Var(obligation.subjectExpr);

      // Check if divisor == 0 is UNSAT (meaning divisor != 0 is always true)
      solver.push();
      solver.add(divisor == 0);

      if (solver.check() == z3::unsat) {
        return ProofResult(true, "Z3 proved divisor is always non-zero");
      }
      solver.pop();
      break;
    }

    case ProofObligation::ArrayBounds: {
      z3::expr index = getZ3Var(obligation.subjectExpr);

      // Try to parse bound as integer or get variable
      long boundValue = 0;
      bool hasBound = false;
      try {
        boundValue = std::stol(obligation.boundExpr);
        hasBound = true;
      } catch (...) {
        // Bound might be a variable
      }

      if (hasBound) {
        // Check if (index < 0 || index >= bound) is UNSAT
        solver.push();
        solver.add(index < 0 || index >= static_cast<int>(boundValue));

        if (solver.check() == z3::unsat) {
          return ProofResult(true, "Z3 proved index is always within bounds");
        }
        solver.pop();
      }
      break;
    }

    default:
      break;
    }

  } catch (const z3::exception &e) {
    return ProofResult(false, std::string("Z3 exception: ") + e.msg());
  }

  return ProofResult(false, "Z3 could not prove the obligation");
}
#endif
