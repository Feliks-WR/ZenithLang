#ifndef ZENITH_IR_IMMUTABILITY_CHECKER_HPP
#define ZENITH_IR_IMMUTABILITY_CHECKER_HPP

#include "zenith/ir/hir.hpp"
#include <string>
#include <vector>

namespace zenith::ir {

// Checks and enforces immutability of parameters in func/pure/math functions
class ImmutabilityChecker {
public:
    struct ImmutabilityError {
        std::string varName;
        std::string message;
        size_t lineNumber = 0;
    };

    // Check a function for immutability violations
    static std::vector<ImmutabilityError> checkFunction(const HirFunc& func);

    // Check if an expression attempts to mutate an immutable variable
    static bool isMutation(const HirExpr& expr);

    // Get all variable references in an expression
    static std::vector<std::string> getVariablesReferenced(const HirExpr& expr);

private:
    static bool isVariableMutation(const std::string& varName, const HirStmt& stmt);
};

}  // namespace zenith::ir

#endif  // ZENITH_IR_IMMUTABILITY_CHECKER_HPP

