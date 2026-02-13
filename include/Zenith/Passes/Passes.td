//===- Passes.td - Zenith passes ---------------------------*- tablegen -*-===//
//
// This file defines the passes for the Zenith dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZENITH_PASSES
#define ZENITH_PASSES

include "mlir/Pass/PassBase.td"

def InlinerPass : Pass<"zenith-inline", "::mlir::ModuleOp"> {
    let summary = "Inline function calls";
    let description = [{
        This pass inlines function calls in the Zenith dialect.
    }];
    let constructor = "mlir::zenith::createInlinerPass()";
}

def ConstantFoldPass : Pass<"zenith-constant-fold", "::mlir::func::FuncOp"> {
    let summary = "Fold constant operations";
    let description = [{
        This pass performs constant folding on Zenith operations.
    }];
    let constructor = "mlir::zenith::createConstantFoldPass()";
}

def ArithmeticOptimizationPass : Pass<"zenith-arith-opt", "::mlir::func::FuncOp"> {
    let summary = "Optimize arithmetic operations";
    let description = [{
        This pass optimizes arithmetic operations in the Zenith dialect,
        including strength reduction and algebraic simplifications.
    }];
    let constructor = "mlir::zenith::createArithmeticOptimizationPass()";
}

def LowerToLLVMPass : Pass<"zenith-lower-to-llvm", "::mlir::ModuleOp"> {
    let summary = "Lower Zenith dialect to LLVM dialect";
    let description = [{
        This pass lowers Zenith dialect operations to LLVM dialect.
    }];
    let constructor = "mlir::zenith::createLowerToLLVMPass()";
    let dependentDialects = ["LLVM::LLVMDialect"];
}

def ShapeInferencePass : Pass<"zenith-shape-inference", "::mlir::func::FuncOp"> {
    let summary = "Perform shape inference";
    let description = [{
        This pass infers shapes for operations in the Zenith dialect.
    }];
    let constructor = "mlir::zenith::createShapeInferencePass()";
}

#endif // ZENITH_PASSES

