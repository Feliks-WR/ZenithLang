#ifndef CUSTOMLANG_PROOFSOLVER_H
#define CUSTOMLANG_PROOFSOLVER_H

#include "Types.h"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace customlang {

struct ProofObligation; // Forward declaration

// Result from attempting to prove an obligation
struct ProofResult {
  bool proved;
  std::string reason;
  std::vector<std::string> facts; // Facts used in proof

  ProofResult(bool p, const std::string &r = "") : proved(p), reason(r) {}
};

// ProofSolver discharges proof obligations using:
// - Constant propagation
// - Constraint inference
// - (Future) SMT solver integration (Z3, CVC5, etc.)
class ProofSolver {
public:
  ProofSolver();

  // Attempt to prove the given obligation
  ProofResult prove(const ProofObligation &obligation, const TypeEnv &typeEnv,
                    const std::unordered_map<std::string, std::optional<long>>
                        &constantValues) const;

private:
  // Try to prove using constant values
  ProofResult
  proveWithConstants(const ProofObligation &obligation,
                     const std::unordered_map<std::string, std::optional<long>>
                         &constantValues) const;

  // Try to prove using type constraints
  ProofResult proveWithTypeConstraints(const ProofObligation &obligation,
                                       const TypeEnv &typeEnv) const;

  // Evaluate expression to constant if possible
  std::optional<long>
  tryEvaluateConstant(const std::string &expr,
                      const std::unordered_map<std::string, std::optional<long>>
                          &constantValues) const;

  // Check if expression is known to be non-zero from type constraints
  bool checkNonZeroFromType(const std::string &expr,
                            const TypeEnv &typeEnv) const;

  // Check if expression is within bounds from type constraints
  bool checkBoundsFromType(const std::string &indexExpr,
                           const std::string &boundExpr,
                           const TypeEnv &typeEnv) const;
};

} // namespace customlang
} // namespace mlir

#endif // CUSTOMLANG_PROOFSOLVER_H
