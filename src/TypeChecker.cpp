#include "TypeChecker.h"
#include <sstream>

using namespace mlir::customlang;

TypeChecker::TypeChecker() {}

void TypeChecker::checkArrayAccess(
    const std::shared_ptr<DependentType> &arrayType,
    const std::string &indexExpr,
    const std::string &location) {
  if (!arrayType || arrayType->kind != DependentType::Array) {
    addError("Type error at " + location + ": array access on non-array type");
    return;
  }

  // Create proof obligation: index must be within bounds
  ProofObligation obligation(ProofObligation::ArrayBounds, location,
                              "Index " + indexExpr + " must be < " +
                                  arrayType->arrayLengthParam);
  addObligation(obligation);
}

void TypeChecker::checkPointerDereference(
    const std::shared_ptr<DependentType> &ptrType,
    const std::string &location) {
  if (!ptrType || ptrType->kind != DependentType::Pointer) {
    addError("Type error at " + location + ": dereference on non-pointer type");
    return;
  }

  // Check if pointer has non-null constraint
  bool hasNonNullConstraint = false;
  for (const auto &constraint : ptrType->constraints) {
    if (constraint->kind == Constraint::NonNull) {
      hasNonNullConstraint = true;
      break;
    }
  }

  if (!hasNonNullConstraint) {
    addWarning("Dereference at " + location + " may fail: pointer could be null");
    ProofObligation obligation(ProofObligation::PointerDeref, location,
                                "Pointer must be non-null");
    addObligation(obligation);
  }
}

void TypeChecker::checkDivision(
    const std::shared_ptr<DependentType> &divisorType,
    const std::string &divisorExpr,
    const std::string &location) {
  if (!divisorType) {
    addError("Type error at " + location + ": unknown divisor type");
    return;
  }

  // Check if divisor has non-zero constraint
  bool hasNonZeroConstraint = false;
  for (const auto &constraint : divisorType->constraints) {
    if (constraint->kind == Constraint::Predicate &&
        (constraint->expression.find("!= 0") != std::string::npos ||
         constraint->expression.find("(!=0)") != std::string::npos)) {
      hasNonZeroConstraint = true;
      break;
    }
  }

  if (!hasNonZeroConstraint) {
    addWarning("Division at " + location + " may fail: divisor could be zero");
    ProofObligation obligation(ProofObligation::DivisionNonZero, location,
                                "Divisor " + divisorExpr + " must != 0");
    obligation.required = Constraint::makePredicate("it != 0");
    addObligation(obligation);
  }
}

void TypeChecker::checkModulo(
    const std::shared_ptr<DependentType> &divisorType,
    const std::string &divisorExpr,
    const std::string &location) {
  if (!divisorType) {
    addError("Type error at " + location + ": unknown modulo divisor type");
    return;
  }

  // Similar to division: check for non-zero constraint
  bool hasNonZeroConstraint = false;
  for (const auto &constraint : divisorType->constraints) {
    if (constraint->kind == Constraint::Predicate &&
        (constraint->expression.find("!= 0") != std::string::npos ||
         constraint->expression.find("(!=0)") != std::string::npos)) {
      hasNonZeroConstraint = true;
      break;
    }
  }

  if (!hasNonZeroConstraint) {
    addWarning("Modulo at " + location + " may fail: divisor could be zero");
    ProofObligation obligation(ProofObligation::ModuloNonZero, location,
                                "Modulo divisor " + divisorExpr + " must != 0");
    obligation.required = Constraint::makePredicate("it != 0");
    addObligation(obligation);
  }
}

void TypeChecker::declareVariable(
    const std::string &name,
    const std::shared_ptr<DependentType> &type) {
  typeEnv.addType(name, type);
}

void TypeChecker::assignVariable(const std::string &name,
                                 const std::string &value) {
  auto type = typeEnv.getType(name);
  if (!type) {
    addError("Undeclared variable: " + name);
    return;
  }

  // Check if value satisfies type constraints
  if (!type->satisfiesConstraints(value)) {
    addError("Value " + value + " does not satisfy constraints for variable " +
             name);
  }
}

std::shared_ptr<DependentType> TypeChecker::getVariableType(
    const std::string &name) const {
  return typeEnv.getType(name);
}

void TypeChecker::addObligation(const ProofObligation &obligation) {
  obligations.push_back(obligation);
}

std::vector<ProofObligation> TypeChecker::getUnsatisfiedObligations() const {
  std::vector<ProofObligation> unsatisfied;
  for (const auto &obligation : obligations) {
    if (!obligation.satisfied) {
      unsatisfied.push_back(obligation);
    }
  }
  return unsatisfied;
}

bool TypeChecker::allObligationsSatisfied() const {
  return getUnsatisfiedObligations().empty();
}

bool TypeChecker::checkConstraint(
    const std::shared_ptr<Constraint> &constraint,
    const std::string &value) const {
  if (!constraint) return false;
  return constraint->isValid();  // Can be extended with actual checking
}

void TypeChecker::addError(const std::string &message) {
  errors.push_back(message);
}

void TypeChecker::addWarning(const std::string &message) {
  warnings.push_back(message);
}

std::string TypeChecker::formatLocation(const std::string &location) const {
  return "[" + location + "]";
}

void TypeChecker::clear() {
  typeEnv.clear();
  obligations.clear();
  errors.clear();
  warnings.clear();
}
