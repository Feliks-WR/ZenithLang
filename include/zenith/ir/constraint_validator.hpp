#ifndef ZENITH_IR_CONSTRAINT_VALIDATOR_HPP
#define ZENITH_IR_CONSTRAINT_VALIDATOR_HPP

#include "zenith/ir/hir.hpp"
#include <optional>
#include <string>
#include <vector>

namespace zenith::ir {

// Validates type constraints and dependent types
class ConstraintValidator {
public:
    // Check if a named constraint is valid (exists in stdlib)
    static bool isValidConstraint(const std::string& constraintName);

    // Validate a type's constraints
    static std::vector<std::string> validateTypeConstraints(const HirType& type);

    // Check if a value is compatible with a type's constraints
    // For literal values, this is compile-time; for variables, this generates runtime checks
    static bool canAssignToType(const HirExpr& value, const HirType& type);

    // Generate runtime assertion code for constraints (if needed)
    static std::string generateConstraintAssertion(const HirType& type, const std::string& varName);
};

}  // namespace zenith::ir

#endif  // ZENITH_IR_CONSTRAINT_VALIDATOR_HPP

