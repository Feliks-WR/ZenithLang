#ifndef CUSTOMLANG_TYPECHECKER_H
#define CUSTOMLANG_TYPECHECKER_H

#include "Types.h"
#include <string>
#include <vector>
#include <memory>

namespace mlir {
namespace customlang {

// Represents a proof obligation that must be satisfied
struct ProofObligation {
  enum Kind {
    ArrayBounds,      // arr[i] where i must be < length
    DivisionNonZero,  // x / y where y must != 0
    ModuloNonZero,    // x % y where y must != 0
    PointerDeref,     // *p where p must be non-null
    PointerValid,     // pointer arithmetic validity
    Custom
  };

  Kind kind;
  std::string location;  // source location (file:line)
  std::string description;
  std::shared_ptr<Constraint> required;
  bool satisfied;

  ProofObligation(Kind k, const std::string &loc, const std::string &desc)
      : kind(k), location(loc), description(desc), satisfied(false) {}
};

// TypeChecker enforces compile-time type constraints and proof obligations
class TypeChecker {
 private:
  TypeEnv typeEnv;
  std::vector<ProofObligation> obligations;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;

 public:
  TypeChecker();

  // Type checking operations
  void checkArrayAccess(const std::shared_ptr<DependentType> &arrayType,
                        const std::string &indexExpr,
                        const std::string &location);
  void checkPointerDereference(const std::shared_ptr<DependentType> &ptrType,
                               const std::string &location);
  void checkDivision(const std::shared_ptr<DependentType> &divisorType,
                     const std::string &divisorExpr,
                     const std::string &location);
  void checkModulo(const std::shared_ptr<DependentType> &divisorType,
                   const std::string &divisorExpr,
                   const std::string &location);

  // Variable and type registration
  void declareVariable(const std::string &name,
                       const std::shared_ptr<DependentType> &type);
  void assignVariable(const std::string &name, const std::string &value);
  std::shared_ptr<DependentType> getVariableType(const std::string &name) const;

  // Proof obligation tracking
  void addObligation(const ProofObligation &obligation);
  std::vector<ProofObligation> getUnsatisfiedObligations() const;
  bool allObligationsSatisfied() const;

  // Constraint satisfaction checking
  bool checkConstraint(const std::shared_ptr<Constraint> &constraint,
                       const std::string &value) const;

  // Error and diagnostic handling
  void addError(const std::string &message);
  void addWarning(const std::string &message);
  const std::vector<std::string> &getErrors() const { return errors; }
  const std::vector<std::string> &getWarnings() const { return warnings; }
  bool hasErrors() const { return !errors.empty(); }

  // Environment management
  TypeEnv &getTypeEnv() { return typeEnv; }
  void clear();

 private:
  std::string formatLocation(const std::string &location) const;
};

}  // namespace customlang
}  // namespace mlir

#endif  // CUSTOMLANG_TYPECHECKER_H
