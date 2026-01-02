#include "zenith/ir/immutability_checker.hpp"
#include <algorithm>
#include <sstream>

namespace zenith::ir {

std::vector<ImmutabilityChecker::ImmutabilityError> ImmutabilityChecker::checkFunction(const HirFunc& func) {
    std::vector<ImmutabilityError> errors;

    // Only check functions with purity constraints
    if (func.purity == HirFunc::Purity::None) {
        return errors;  // proc and subroutine don't enforce immutability
    }

    // Collect all immutable parameter names
    std::vector<std::string> immutableVars;
    for (const auto& param : func.params) {
        if (param.isImmutable) {
            immutableVars.push_back(param.name);
        }
    }

    // Check all statements for mutations of immutable variables
    for (const auto& stmt : func.body) {
        if (stmt.kind == HirStmt::Kind::Assign) {
            // Check if trying to assign to an immutable parameter
            auto it = std::find(immutableVars.begin(), immutableVars.end(), stmt.name);
            if (it != immutableVars.end()) {
                ImmutabilityError err;
                err.varName = stmt.name;
                err.message = "Cannot assign to immutable parameter '" + stmt.name + "' in " +
                             (func.purity == HirFunc::Purity::Math ? "math" :
                              func.purity == HirFunc::Purity::Pure ? "pure" : "func") +
                             " function '" + func.name + "'";
                errors.push_back(err);
            }
        }
    }

    return errors;
}

bool ImmutabilityChecker::isMutation(const HirExpr& expr) {
    // An expression is a mutation if it's an assignment-like operation
    // This is handled at the statement level (HirStmt::Kind::Assign)
    // Expressions themselves don't perform mutations in this design
    return false;
}

std::vector<std::string> ImmutabilityChecker::getVariablesReferenced(const HirExpr& expr) {
    std::vector<std::string> vars;

    switch (expr.kind) {
        case HirExpr::Kind::Id:
            vars.push_back(expr.strVal);
            break;
        case HirExpr::Kind::BinOp:
            if (expr.lhs) {
                auto lhsVars = getVariablesReferenced(*expr.lhs);
                vars.insert(vars.end(), lhsVars.begin(), lhsVars.end());
            }
            if (expr.rhs) {
                auto rhsVars = getVariablesReferenced(*expr.rhs);
                vars.insert(vars.end(), rhsVars.begin(), rhsVars.end());
            }
            break;
        case HirExpr::Kind::Array:
            for (const auto& elem : expr.elements) {
                auto elemVars = getVariablesReferenced(elem);
                vars.insert(vars.end(), elemVars.begin(), elemVars.end());
            }
            break;
        case HirExpr::Kind::Call:
            for (const auto& elem : expr.elements) {
                auto elemVars = getVariablesReferenced(elem);
                vars.insert(vars.end(), elemVars.begin(), elemVars.end());
            }
            break;
        case HirExpr::Kind::Group:
        case HirExpr::Kind::Unsafe:
            if (expr.lhs) {
                auto innerVars = getVariablesReferenced(*expr.lhs);
                vars.insert(vars.end(), innerVars.begin(), innerVars.end());
            }
            break;
        default:
            break;
    }

    return vars;
}

bool ImmutabilityChecker::isVariableMutation(const std::string& varName, const HirStmt& stmt) {
    if (stmt.kind == HirStmt::Kind::Assign && stmt.name == varName) {
        return true;
    }
    return false;
}

}  // namespace zenith::ir

