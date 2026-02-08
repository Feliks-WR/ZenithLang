#include "Types.h"

#include <algorithm>
#include <sstream>
#include <utility>

using namespace mlir::customlang;

// ============================================================================
// Constraint Implementation
// ============================================================================

Constraint::Constraint(const ConstraintKind Kind, std::string Expr)
    : kind(Kind), expression(std::move(Expr)), minValue(0), maxValue(0) {}

std::shared_ptr<Constraint> Constraint::makeRange(const long Min,
                                                  const long Max) {
  auto constraint = std::make_shared<Constraint>(Constraint::Range, "");
  constraint->minValue = Min;
  constraint->maxValue = Max;
  constraint->expression = std::to_string(Min) + ".." + std::to_string(Max);
  return constraint;
}

std::shared_ptr<Constraint> Constraint::makeSingleValue(const long Value) {
  auto constraint = std::make_shared<Constraint>(SingleValue, "");
  constraint->expression = std::to_string(Value);
  return constraint;
}

std::shared_ptr<Constraint> Constraint::makePredicate(const std::string &Expr) {
  return std::make_shared<Constraint>(Predicate, Expr);
}

std::shared_ptr<Constraint> Constraint::makeNonNull() {
  return std::make_shared<Constraint>(NonNull, "nonnull");
}

std::string Constraint::toString() const {
  switch (kind) {
  case Range:
  case SingleValue:
  case Predicate:
    return "{" + expression + "}";
  case NonNull:
    return "{nonnull}";
  case Custom:
    return "{" + expression + "}";
  default:
    return "{}";
  }
}

bool Constraint::isValid() const { return !expression.empty(); }

// ============================================================================
// DependentType Implementation
// ============================================================================

DependentType::DependentType() : kind(Named), baseName("unknown") {}

DependentType::DependentType(const TypeKind Kind, std::string Name)
    : kind(Kind), baseName(std::move(Name)) {}

std::shared_ptr<DependentType> DependentType::makeInt() {
  return std::make_shared<DependentType>(Int, "int");
}

std::shared_ptr<DependentType> DependentType::makeIntWithConstraint(
    const std::shared_ptr<Constraint> &Constraint) {
  auto Type = makeInt();
  Type->constraints.push_back(Constraint);
  return Type;
}

std::shared_ptr<DependentType>
DependentType::makePointer(const std::shared_ptr<DependentType> &ElemType) {
  auto Type = std::make_shared<DependentType>(Pointer, "ptr");
  Type->elementType = ElemType;
  return Type;
}

std::shared_ptr<DependentType>
DependentType::makeArray(const std::shared_ptr<DependentType> &ElemType,
                         const std::string &LengthParam) {
  auto Type = std::make_shared<DependentType>(Array, "array");
  Type->arrayElementType = ElemType;
  Type->arrayLengthParam = LengthParam;
  return Type;
}

std::shared_ptr<DependentType>
DependentType::makeSet(const std::shared_ptr<DependentType> &ElemType) {
  auto Type = std::make_shared<DependentType>(Set, "set");
  Type->setElementType = ElemType;
  return Type;
}

std::shared_ptr<DependentType> DependentType::makeFloat() {
  return std::make_shared<DependentType>(Float, "float");
}

std::shared_ptr<DependentType> DependentType::makeBool() {
  return std::make_shared<DependentType>(Bool, "bool");
}

std::shared_ptr<DependentType>
DependentType::makeNamed(const std::string &Name) {
  return std::make_shared<DependentType>(Named, Name);
}

bool DependentType::requiresProof() const { return !constraints.empty(); }

std::vector<std::string> DependentType::getProofObligations() const {
  std::vector<std::string> Proofs(constraints.size());
  for (const auto &Constraint : constraints) {
    Proofs.push_back(Constraint->toString());
  }
  return Proofs;
}

std::string DependentType::toString() const {
  std::stringstream Ss;

  switch (kind) {
  case Int:
    Ss << "int";
    break;
  case Float:
    Ss << "float";
    break;
  case Bool:
    Ss << "bool";
    break;
  case Pointer:
    Ss << "*";
    if (elementType) {
      Ss << elementType->toString();
    }
    break;
  case Array:
    Ss << "[" << (arrayElementType ? arrayElementType->toString() : "?");
    Ss << "; " << arrayLengthParam << "]";
    break;
  case Set:
    Ss << "{";
    if (setElementType) {
      Ss << setElementType->toString();
    }
    Ss << "}";
    break;
  case Function:
    Ss << "(";
    for (size_t I = 0; I < paramTypes.size(); ++I) {
      if (I > 0)
        Ss << ", ";
      Ss << paramTypes[I]->toString();
    }
    Ss << ") -> ";
    Ss << (returnType ? returnType->toString() : "void");
    break;
  case Named:
    Ss << baseName;
    break;
  }

  // Append constraints
  for (const auto &Constraint : constraints) {
    Ss << Constraint->toString();
  }

  return Ss.str();
}

bool DependentType::isCompatibleWith(
    const std::shared_ptr<DependentType> &Other) const {
  if (!Other)
    return false;

  // Base type must match
  if (kind != Other->kind || baseName != Other->baseName) {
    return false;
  }

  // For pointers, element types must be compatible
  if (kind == Pointer) {
    if (!elementType || !Other->elementType) {
      return elementType == Other->elementType;
    }
    return elementType->isCompatibleWith(Other->elementType);
  }

  // For arrays, element types and length must match
  if (kind == Array) {
    if (!arrayElementType || !Other->arrayElementType) {
      return arrayElementType == Other->arrayElementType;
    }
    return arrayElementType->isCompatibleWith(Other->arrayElementType) &&
           arrayLengthParam == Other->arrayLengthParam;
  }

  // For sets, element types must be compatible
  if (kind == Set) {
    if (!setElementType || !Other->setElementType) {
      return setElementType == Other->setElementType;
    }
    return setElementType->isCompatibleWith(Other->setElementType);
  }

  return true;
}

bool DependentType::satisfiesConstraints(const std::string &Value) const {
  return std::all_of(constraints.begin(), constraints.end(),
                     [&Value](const std::shared_ptr<Constraint> &Constraint) {
                       if (Constraint->kind == Constraint::Range) {
                         try {
                           if (const long Val = std::stol(Value);
                               Val < Constraint->minValue ||
                               Val > Constraint->maxValue) {
                             return false;
                           }
                         } catch (...) {
                           return false;
                         }
                       } else if (Constraint->kind == Constraint::SingleValue) {
                         if (Constraint->expression != Value) {
                           return false;
                         }
                       } else if (Constraint->kind == Constraint::Predicate) {
                         try {
                           const long Val = std::stol(Value);
                           const std::string &Expr = Constraint->expression;
                           if (Expr.find("!= 0") != std::string::npos ||
                               Expr.find("(!=0)") != std::string::npos ||
                               Expr.find("it != 0") != std::string::npos) {
                             if (Val == 0)
                               return false;
                           }
                           if (Expr.find("it > 0") != std::string::npos ||
                               Expr.find("(>0)") != std::string::npos) {
                             if (Val <= 0)
                               return false;
                           }
                           if (Expr.find("it >= 1") != std::string::npos ||
                               Expr.find("(>=1)") != std::string::npos) {
                             if (Val < 1)
                               return false;
                           }
                           if (Expr.find("it < 0") != std::string::npos ||
                               Expr.find("(<0)") != std::string::npos) {
                             if (Val >= 0)
                               return false;
                           }
                           if (Expr.find("it <= -1") != std::string::npos ||
                               Expr.find("(<=-1)") != std::string::npos) {
                             if (Val > -1)
                               return false;
                           }
                         } catch (...) {
                           return false;
                         }
                       }
                       return true;
                     });
}

// ============================================================================
// TypeEnv Implementation
// ============================================================================

void TypeEnv::addType(const std::string &Name,
                      const std::shared_ptr<DependentType> &Type) {
  types[Name] = Type;
}

std::shared_ptr<DependentType> TypeEnv::getType(const std::string &Name) const {
  if (const auto It = types.find(Name); It != types.end()) {
    return It->second;
  }
  return nullptr;
}

void TypeEnv::addProofObligation(const std::string &Location,
                                 const std::shared_ptr<Constraint> &Proof) {
  proofObligations[Location].push_back(Proof);
}

std::vector<std::shared_ptr<Constraint>>
TypeEnv::getProofs(const std::string &Location) const {
  if (const auto It = proofObligations.find(Location);
      It != proofObligations.end()) {
    return It->second;
  }
  return {};
}

bool TypeEnv::hasType(const std::string &Name) const {
  return types.find(Name) != types.end();
}

void TypeEnv::clear() {
  types.clear();
  proofObligations.clear();
}
