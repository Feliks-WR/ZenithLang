#include "Types.h"
#include <sstream>

using namespace mlir::customlang;

// ============================================================================
// Constraint Implementation
// ============================================================================

Constraint::Constraint(ConstraintKind k, const std::string &expr)
    : kind(k), expression(expr), minValue(0), maxValue(0) {}

std::shared_ptr<Constraint> Constraint::makeRange(long min, long max) {
  auto constraint = std::make_shared<Constraint>(Range, "");
  constraint->minValue = min;
  constraint->maxValue = max;
  constraint->expression = std::to_string(min) + ".." + std::to_string(max);
  return constraint;
}

std::shared_ptr<Constraint> Constraint::makePredicate(
    const std::string &expr) {
  return std::make_shared<Constraint>(Predicate, expr);
}

std::shared_ptr<Constraint> Constraint::makeNonNull() {
  return std::make_shared<Constraint>(NonNull, "nonnull");
}

std::string Constraint::toString() const {
  switch (kind) {
    case Range:
      return "{" + expression + "}";
    case SingleValue:
      return "{" + expression + "}";
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

bool Constraint::isValid() const {
  return !expression.empty();
}

// ============================================================================
// DependentType Implementation
// ============================================================================

DependentType::DependentType()
    : kind(Named), baseName("unknown") {}

DependentType::DependentType(TypeKind k, const std::string &name)
    : kind(k), baseName(name) {}

std::shared_ptr<DependentType> DependentType::makeInt() {
  return std::make_shared<DependentType>(Int, "int");
}

std::shared_ptr<DependentType> DependentType::makeIntWithConstraint(
    const std::shared_ptr<Constraint> &constraint) {
  auto type = makeInt();
  type->constraints.push_back(constraint);
  return type;
}

std::shared_ptr<DependentType> DependentType::makePointer(
    const std::shared_ptr<DependentType> &elemType) {
  auto type = std::make_shared<DependentType>(Pointer, "ptr");
  type->elementType = elemType;
  return type;
}

std::shared_ptr<DependentType> DependentType::makeArray(
    const std::shared_ptr<DependentType> &elemType,
    const std::string &lengthParam) {
  auto type = std::make_shared<DependentType>(Array, "array");
  type->arrayElementType = elemType;
  type->arrayLengthParam = lengthParam;
  return type;
}

std::shared_ptr<DependentType> DependentType::makeFloat() {
  return std::make_shared<DependentType>(Float, "float");
}

std::shared_ptr<DependentType> DependentType::makeBool() {
  return std::make_shared<DependentType>(Bool, "bool");
}

std::shared_ptr<DependentType> DependentType::makeNamed(
    const std::string &name) {
  return std::make_shared<DependentType>(Named, name);
}

bool DependentType::requiresProof() const {
  return !constraints.empty();
}

std::vector<std::string> DependentType::getProofObligations() const {
  std::vector<std::string> proofs;
  for (const auto &constraint : constraints) {
    proofs.push_back(constraint->toString());
  }
  return proofs;
}

std::string DependentType::toString() const {
  std::stringstream ss;
  
  switch (kind) {
    case Int:
      ss << "int";
      break;
    case Float:
      ss << "float";
      break;
    case Bool:
      ss << "bool";
      break;
    case Pointer:
      ss << "*";
      if (elementType) {
        ss << elementType->toString();
      }
      break;
    case Array:
      ss << "[" << (arrayElementType ? arrayElementType->toString() : "?");
      ss << "; " << arrayLengthParam << "]";
      break;
    case Function:
      ss << "(";
      for (size_t i = 0; i < paramTypes.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << paramTypes[i]->toString();
      }
      ss << ") -> ";
      ss << (returnType ? returnType->toString() : "void");
      break;
    case Named:
      ss << baseName;
      break;
  }
  
  // Append constraints
  for (const auto &constraint : constraints) {
    ss << constraint->toString();
  }
  
  return ss.str();
}

bool DependentType::isCompatibleWith(
    const std::shared_ptr<DependentType> &other) const {
  if (!other) return false;
  
  // Base type must match
  if (kind != other->kind || baseName != other->baseName) {
    return false;
  }
  
  // For pointers, element types must be compatible
  if (kind == Pointer) {
    if (!elementType || !other->elementType) {
      return elementType == other->elementType;
    }
    return elementType->isCompatibleWith(other->elementType);
  }
  
  // For arrays, element types and length must match
  if (kind == Array) {
    if (!arrayElementType || !other->arrayElementType) {
      return arrayElementType == other->arrayElementType;
    }
    return arrayElementType->isCompatibleWith(other->arrayElementType) &&
           arrayLengthParam == other->arrayLengthParam;
  }
  
  return true;
}

bool DependentType::satisfiesConstraints(const std::string &value) const {
  for (const auto &constraint : constraints) {
    if (constraint->kind == Constraint::Range) {
      try {
        long val = std::stol(value);
        if (val < constraint->minValue || val > constraint->maxValue) {
          return false;
        }
      } catch (...) {
        return false;
      }
    } else if (constraint->kind == Constraint::Predicate) {
      if (constraint->expression.find("!= 0") != std::string::npos ||
          constraint->expression.find("(!=0)") != std::string::npos) {
        try {
          long val = std::stol(value);
          if (val == 0) return false;
        } catch (...) {
          return false;
        }
      }
    }
  }
  return true;
}

// ============================================================================
// TypeEnv Implementation
// ============================================================================

void TypeEnv::addType(const std::string &name,
                      const std::shared_ptr<DependentType> &type) {
  types[name] = type;
}

std::shared_ptr<DependentType> TypeEnv::getType(const std::string &name) const {
  auto it = types.find(name);
  if (it != types.end()) {
    return it->second;
  }
  return nullptr;
}

void TypeEnv::addProofObligation(const std::string &location,
                                 const std::shared_ptr<Constraint> &proof) {
  proofObligations[location].push_back(proof);
}

std::vector<std::shared_ptr<Constraint>> TypeEnv::getProofs(
    const std::string &location) const {
  auto it = proofObligations.find(location);
  if (it != proofObligations.end()) {
    return it->second;
  }
  return {};
}

bool TypeEnv::hasType(const std::string &name) const {
  return types.find(name) != types.end();
}

void TypeEnv::clear() {
  types.clear();
  proofObligations.clear();
}
