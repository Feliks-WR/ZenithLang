#ifndef CUSTOMLANG_TYPES_H
#define CUSTOMLANG_TYPES_H

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace customlang {

// Constraint represents compile-time proof obligations
// Examples: {it != 0}, {(!=0)}, {nonnull}, etc.
class Constraint {
public:
  enum ConstraintKind {
    Range,       // int {1..10}
    SingleValue, // int {5}
    Predicate,   // int {it != 0} or int {(!=0)}
    NonNull,     // *T {nonnull}
    Custom
  };

  ConstraintKind kind;
  std::string expression; // e.g., "it != 0", "(!=0)", "nonnull"

  // For range constraints
  long minValue;
  long maxValue;

  Constraint(ConstraintKind Kind, std::string Expr);
  static std::shared_ptr<Constraint> makeRange(long Min, long Max);
  static std::shared_ptr<Constraint> makeSingleValue(long Value);
  static std::shared_ptr<Constraint> makePredicate(const std::string &Expr);
  static std::shared_ptr<Constraint> makeNonNull();

  std::string toString() const;
  bool isValid() const;
};

// DependentType represents types that depend on values
// Examples: int {it != 0}, [T; N], *int, etc.
class DependentType {
public:
  enum TypeKind { Int, Float, Bool, Pointer, Array, Set, Function, Named };

  TypeKind kind;
  std::string baseName; // "int", "float", "bool", custom names

  // For pointers: element type
  std::shared_ptr<DependentType> elementType;

  // For arrays: element type and length parameter
  std::shared_ptr<DependentType> arrayElementType;
  std::string arrayLengthParam; // e.g., "N"

  // For sets: element type
  std::shared_ptr<DependentType> setElementType;

  // Dependent type constraints
  std::vector<std::shared_ptr<Constraint>> constraints;

  // For function types: parameter and return types
  std::vector<std::shared_ptr<DependentType>> paramTypes;
  std::shared_ptr<DependentType> returnType;

  DependentType();
  DependentType(TypeKind Kind, std::string Name);

  // Factory methods
  static std::shared_ptr<DependentType> makeInt();
  static std::shared_ptr<DependentType>
  makeIntWithConstraint(const std::shared_ptr<Constraint> &Constraint);
  static std::shared_ptr<DependentType>
  makePointer(const std::shared_ptr<DependentType> &ElemType);
  static std::shared_ptr<DependentType>
  makeArray(const std::shared_ptr<DependentType> &ElemType,
            const std::string &LengthParam);
  static std::shared_ptr<DependentType>
  makeSet(const std::shared_ptr<DependentType> &ElemType);
  static std::shared_ptr<DependentType> makeFloat();
  static std::shared_ptr<DependentType> makeBool();
  static std::shared_ptr<DependentType> makeNamed(const std::string &Name);

  // Proof checking
  bool requiresProof() const;
  std::vector<std::string> getProofObligations() const;

  // String representation
  std::string toString() const;

  // Type compatibility
  bool isCompatibleWith(const std::shared_ptr<DependentType> &Other) const;

  // Constraint satisfaction
  bool satisfiesConstraints(const std::string &Value) const;
};

// Type environment for type checking
class TypeEnv {
private:
  std::map<std::string, std::shared_ptr<DependentType>> types;
  std::map<std::string, std::vector<std::shared_ptr<Constraint>>>
      proofObligations;

public:
  void addType(const std::string &Name,
               const std::shared_ptr<DependentType> &Type);
  std::shared_ptr<DependentType> getType(const std::string &Name) const;

  void addProofObligation(const std::string &Location,
                          const std::shared_ptr<Constraint> &Proof);
  std::vector<std::shared_ptr<Constraint>>
  getProofs(const std::string &Location) const;

  bool hasType(const std::string &Name) const;
  void clear();
};

} // namespace customlang
} // namespace mlir

#endif // CUSTOMLANG_TYPES_H
