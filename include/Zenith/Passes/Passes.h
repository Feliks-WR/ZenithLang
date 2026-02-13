//===- Passes.h - Zenith Passes --------------------------------*- C++ -*-===//
//
// This file defines passes for the Zenith dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZENITH_PASSES_H
#define ZENITH_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace zenith {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Create a pass that inlines function calls.
std::unique_ptr<Pass> createInlinerPass();

/// Create a pass that performs constant folding.
std::unique_ptr<Pass> createConstantFoldPass();

/// Create a pass that optimizes arithmetic operations.
std::unique_ptr<Pass> createArithmeticOptimizationPass();

/// Create a pass that lowers Zenith dialect to LLVM dialect.
std::unique_ptr<Pass> createLowerToLLVMPass();

/// Create a pass that performs shape inference.
std::unique_ptr<Pass> createShapeInferencePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Zenith/Passes/Passes.h.inc"

} // namespace zenith
} // namespace mlir

#endif // ZENITH_PASSES_H

