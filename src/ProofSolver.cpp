#include "ProofSolver.h"
#include "TypeChecker.h"
#include <algorithm>
#include <cctype>
#include <sstream>

using namespace mlir::customlang;

ProofSolver::ProofSolver() {}

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
