//===- zenith-opt.cpp - Zenith optimizer driver ---------------------------===//
//
// This file implements the 'zenith-opt' tool, which is the Zenith compiler's
// optimizer driver.
//
//===----------------------------------------------------------------------===//

#include "Zenith/Dialect/ZenithDialect.h"
#include "Zenith/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

int main(int argc, char **argv) {
    mlir::registerAllPasses();
    mlir::zenith::registerPasses();

    mlir::DialectRegistry registry;
    registry.insert<mlir::zenith::ZenithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    mlir::registerAllDialects(registry);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Zenith optimizer driver\n", registry));
}

