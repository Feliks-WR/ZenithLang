#ifndef CUSTOMLANG_TYPES_H
#define CUSTOMLANG_TYPES_H

#include <string>
#include <vector>
#include <memory>
#include <map>

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
  std::string expression;  // e.g., "it != 0", "(!=0)", "nonnull"
  
  // For range constraints
  long minValue;
  long maxValue;

  Constraint(ConstraintKind k, const std::string &expr);
  
  static std::shared_ptr<Constraint> makeRange(long min, long max);
  static std::shared_ptr<Constraint> makeSingleValue(long value);
  static std::shared_ptr<Constraint> makePredicate(const std::string &expr);
  static std::shared_ptr<Constraint> makeNonNull();
  
  std::string toString() const;
  bool isValid() const;
};

// DependentType represents types that depend on values
// Examples: int {it != 0}, [T; N], *int, etc.
class DependentType {
 public:
  enum TypeKind {
    Int,
    Float,
    Bool,
    Pointer,
    Array,
    Set,
    Function,
    Named
  };

  TypeKind kind;
  std::string baseName;  // "int", "float", "bool", custom names
  
  // For pointers: element type
  std::shared_ptr<DependentType> elementType;
  
  // For arrays: element type and length parameter
  std::shared_ptr<DependentType> arrayElementType;
  std::string arrayLengthParam;  // e.g., "N"
  
  // For sets: element type
  std::shared_ptr<DependentType> setElementType;
  
  // Dependent type constraints
  std::vector<std::shared_ptr<Constraint>> constraints;
  
  // For function types: parameter and return types
  std::vector<std::shared_ptr<DependentType>> paramTypes;
  std::shared_ptr<DependentType> returnType;

  DependentType();
  DependentType(TypeKind k, const std::string &name);

  // Factory methods
  static std::shared_ptr<DependentType> makeInt();
  static std::shared_ptr<DependentType> makeIntWithConstraint(
      const std::shared_ptr<Constraint> &constraint);
  static std::shared_ptr<DependentType> makePointer(
      const std::shared_ptr<DependentType> &elemType);
  static std::shared_ptr<DependentType> makeArray(
      const std::shared_ptr<DependentType> &elemType,
      const std::string &lengthParam);
  static std::shared_ptr<DependentType> makeSet(
      const std::shared_ptr<DependentType> &elemType);
  static std::shared_ptr<DependentType> makeFloat();
  static std::shared_ptr<DependentType> makeBool();
  static std::shared_ptr<DependentType> makeNamed(const std::string &name);

  // Proof checking
  bool requiresProof() const;
  std::vector<std::string> getProofObligations() const;

  // String representation
  std::string toString() const;
  
  // Type compatibility
  bool isCompatibleWith(const std::shared_ptr<DependentType> &other) const;
  
  // Constraint satisfaction
  bool satisfiesConstraints(const std::string &value) const;
};

// Type environment for type checking
class TypeEnv {
 private:
  std::map<std::string, std::shared_ptr<DependentType>> types;
  std::map<std::string, std::vector<std::shared_ptr<Constraint>>> 
      proofObligations;

 public:
  void addType(const std::string &name, 
               const std::shared_ptr<DependentType> &type);
  std::shared_ptr<DependentType> getType(const std::string &name) const;
  
  void addProofObligation(const std::string &location,
                          const std::shared_ptr<Constraint> &proof);
  std::vector<std::shared_ptr<Constraint>> getProofs(
      const std::string &location) const;
  
  bool hasType(const std::string &name) const;
  void clear();
};

}  // namespace customlang
}  // namespace mlir

#endif  // CUSTOMLANG_TYPES_H
