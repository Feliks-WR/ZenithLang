#include "zenith/ir/constraint_validator.hpp"
#include "zenith/stdlib_registry.hpp"
#include <algorithm>
#include <sstream>
#include <cmath>

namespace zenith::ir {

bool ConstraintValidator::isValidConstraint(const std::string& constraintName) {
    return stdlib::StdlibRegistry::instance().isConstraintDefined(constraintName);
}

std::vector<std::string> ConstraintValidator::validateTypeConstraints(const HirType& type) {
    std::vector<std::string> errors;

    // Check all named constraints are valid
    for (const auto& constraint : type.namedConstraints) {
        if (!isValidConstraint(constraint)) {
            errors.push_back("Unknown constraint: " + constraint);
        }
    }

    // Validate range constraints (basic range checks)
    if (type.intMin.has_value() && type.intMax.has_value()) {
        if (type.intMin.value() > type.intMax.value()) {
            errors.push_back("Invalid range: min > max");
        }
    }

    return errors;
}

bool ConstraintValidator::canAssignToType(const HirExpr& value, const HirType& type) {
    // For now, a simple implementation
    // In a full implementation, this would:
    // 1. Check literal values against constraints at compile-time
    // 2. Mark variable assignments for runtime checking

    if (value.kind == HirExpr::Kind::IntLit) {
        // Check range constraints for literal integers
        if (type.intMin.has_value() && value.intVal < type.intMin.value()) {
            return false;
        }
        if (type.intMax.has_value() && value.intVal > type.intMax.value()) {
            return false;
        }

        // Check "even" constraint
        if (std::find(type.namedConstraints.begin(), type.namedConstraints.end(), "even") !=
            type.namedConstraints.end()) {
            if (value.intVal % 2 != 0) return false;
        }

        // Check "positive" constraint
        if (std::find(type.namedConstraints.begin(), type.namedConstraints.end(), "positive") !=
            type.namedConstraints.end()) {
            if (value.intVal <= 0) return false;
        }

        return true;
    }

    if (value.kind == HirExpr::Kind::FloatLit) {
        // Check "not_nan" constraint
        if (std::find(type.namedConstraints.begin(), type.namedConstraints.end(), "not_nan") !=
            type.namedConstraints.end()) {
            if (std::isnan(value.floatVal)) return false;
        }

        // Check "not_inf" constraint
        if (std::find(type.namedConstraints.begin(), type.namedConstraints.end(), "not_inf") !=
            type.namedConstraints.end()) {
            if (std::isinf(value.floatVal)) return false;
        }

        return true;
    }

    // For variable assignments, we accept them (runtime checking will be added)
    return true;
}

std::string ConstraintValidator::generateConstraintAssertion(const HirType& type, const std::string& varName) {
    std::ostringstream oss;

    // Generate runtime assertions for constraints
    if (type.intMin.has_value()) {
        oss << "assert(" << varName << " >= " << type.intMin.value() << ");\n";
    }
    if (type.intMax.has_value()) {
        oss << "assert(" << varName << " <= " << type.intMax.value() << ");\n";
    }

    for (const auto& constraint : type.namedConstraints) {
        if (constraint == "positive") {
            oss << "assert(" << varName << " > 0);\n";
        } else if (constraint == "negative") {
            oss << "assert(" << varName << " < 0);\n";
        } else if (constraint == "even") {
            oss << "assert(" << varName << " % 2 == 0);\n";
        } else if (constraint == "odd") {
            oss << "assert(" << varName << " % 2 != 0);\n";
        } else if (constraint == "not_nan") {
            oss << "assert(!isnan(" << varName << "));\n";
        } else if (constraint == "not_inf") {
            oss << "assert(!isinf(" << varName << "));\n";
        }
    }

    return oss.str();
}

}  // namespace zenith::ir

