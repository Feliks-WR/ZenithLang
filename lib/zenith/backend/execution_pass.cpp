#include "zenith/backend/execution_pass.hpp"
#include "zenith/ir/hir.hpp"
#include <functional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace zenith::backend {

void ExecutionPass::run(const ir::HirModule& hir) {
    using namespace mlir;

    if (debug_) {
        llvm::errs() << "=== Execution Pass ===\n";
        llvm::errs() << "Parsed " << hir.topLevel.size() << " top-level statements\n";
        llvm::errs() << "Parsed " << hir.functions.size() << " functions\n\n";
    }

    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Create MLIR context and load dialects
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect>();
    ctx.loadDialect<arith::ArithDialect>();
    ctx.loadDialect<memref::MemRefDialect>();
    ctx.loadDialect<LLVM::LLVMDialect>();

    OpBuilder builder(&ctx);
    auto mlirModule = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(mlirModule.getBody());

    // String counter for unique global string names
    int stringCounter = 0;

    // Lambda: lower expression to MLIR value
    std::function<Value(const ir::HirExpr&, OpBuilder&, mlir::Block*)> lowerExpr;
    lowerExpr = [&](const ir::HirExpr& e, OpBuilder& b, mlir::Block* blk) -> Value {
        b.setInsertionPointToEnd(blk);
        switch (e.kind) {
            case ir::HirExpr::Kind::IntLit: {
                auto ty = b.getI64Type();
                return b.create<arith::ConstantOp>(
                    b.getUnknownLoc(), ty, b.getI64IntegerAttr(e.intVal));
            }
            case ir::HirExpr::Kind::FloatLit: {
                auto ty = b.getF64Type();
                return b.create<arith::ConstantOp>(
                    b.getUnknownLoc(), ty, b.getF64FloatAttr(e.floatVal));
            }
            case ir::HirExpr::Kind::BoolLit: {
                auto ty = b.getI1Type();
                return b.create<arith::ConstantOp>(
                    b.getUnknownLoc(), ty, b.getBoolAttr(e.boolVal));
            }
            case ir::HirExpr::Kind::StringLit: {
                // Create an immutable global string
                std::string globalName = "zenith_str_" + std::to_string(stringCounter++);
                const size_t strLen = e.strVal.length();

                SmallVector<int8_t> strBytes;
                for (char c : e.strVal) {
                    strBytes.push_back(static_cast<int8_t>(c));
                }

                [[maybe_unused]] OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(mlirModule.getBody());

                const auto i8Type = b.getI8Type();
                const auto memrefType = MemRefType::get({static_cast<int64_t>(strLen)}, i8Type);
                const auto tensorType = RankedTensorType::get({static_cast<int64_t>(strLen)}, i8Type);
                const auto initialValue
                    = DenseElementsAttr::get(tensorType, llvm::ArrayRef(strBytes));

                b.create<memref::GlobalOp>(
                    b.getUnknownLoc(), globalName, b.getStringAttr("private"),
                    memrefType, initialValue, /*constant=*/true, /*alignment=*/nullptr
                );

                return nullptr;
            }
            case ir::HirExpr::Kind::Array: {
                SmallVector<Value> elemVals;
                Type elemType = nullptr;

                for (const auto& elem : e.elements) {
                    if (const auto val = lowerExpr(elem, b, blk)) {
                        elemVals.push_back(val);
                        if (!elemType) elemType = val.getType();
                    }
                }

                if (elemVals.empty()) return nullptr;

                auto arraySize = static_cast<int64_t>(elemVals.size());
                auto memrefType = MemRefType::get({arraySize}, elemType);
                auto allocaOp = b.create<memref::AllocaOp>(b.getUnknownLoc(), memrefType);

                for (size_t i = 0; i < elemVals.size(); ++i) {
                    auto indexVal = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), i);
                    b.create<memref::StoreOp>(b.getUnknownLoc(), elemVals[i], allocaOp, ValueRange{indexVal});
                }

                return allocaOp.getResult();
            }
            case ir::HirExpr::Kind::Id:
            case ir::HirExpr::Kind::Call:
                return nullptr;
            case ir::HirExpr::Kind::Group:
            case ir::HirExpr::Kind::Unsafe:
                if (!e.elements.empty()) return lowerExpr(e.elements[0], b, blk);
                return nullptr;
            case ir::HirExpr::Kind::BinOp: {
                if (!e.lhs || !e.rhs) return nullptr;

                Value lhsVal = lowerExpr(*e.lhs, b, blk);
                Value rhsVal = lowerExpr(*e.rhs, b, blk);

                if (!lhsVal || !rhsVal) return nullptr;

                switch (e.binOpKind) {
                    case ir::HirExpr::BinOpKind::Add: {
                        if (lhsVal.getType().isInteger(64)) {
                            return b.create<arith::AddIOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        } else {
                            return b.create<arith::AddFOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        }
                    }
                    case ir::HirExpr::BinOpKind::Sub: {
                        if (lhsVal.getType().isInteger(64))
                            return b.create<arith::SubIOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        return b.create<arith::SubFOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                    }
                    case ir::HirExpr::BinOpKind::Mul: {
                        if (lhsVal.getType().isInteger(64)) {
                            return b.create<arith::MulIOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        }
                        return b.create<arith::MulFOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                    }
                    case ir::HirExpr::BinOpKind::Div: {
                        if (lhsVal.getType().isInteger(64)) {
                            return b.create<arith::DivSIOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        } else {
                            return b.create<arith::DivFOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        }
                    }
                    case ir::HirExpr::BinOpKind::Power: {
                        // For power operation, we use a simple implementation
                        // For float types, this would need to call a math library function
                        // For now, we'll handle it as a TODO
                        // TODO: Implement proper power operation using math library
                        return nullptr;
                    }
                    case ir::HirExpr::BinOpKind::Concat: {
                        // String/array concatenation would require additional implementation
                        // For now, return nullptr as it requires string manipulation ops
                        return nullptr;
                    }
                }
                return nullptr;
            }
        }
        return nullptr;
    };

    // Lambda: lower statement
    const std::function lowerStmt = [&](const ir::HirStmt& stmt, OpBuilder& b, mlir::Block* blk) {
        b.setInsertionPointToEnd(blk);
        switch (stmt.kind) {
            case ir::HirStmt::Kind::Assign: {
                lowerExpr(stmt.expr, b, blk);
                break;
            }
            case ir::HirStmt::Kind::Call: {
                break;
            }
        }
    };

    // Lower top-level statements into a main function
    auto mainFnType = builder.getFunctionType({}, builder.getI32Type());
    auto mainFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(), "main", mainFnType);
    auto& mainBody = mainFunc.getBody();
    mlir::Block* mainBlock = &mainBody.emplaceBlock();

    builder.setInsertionPointToEnd(mainBlock);

    for (const auto& stmt : hir.topLevel) {
        lowerStmt(stmt, builder, mainBlock);
    }

    // Return 0 from main
    auto zeroVal = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange{zeroVal.getResult()});

    if (debug_) {
        llvm::errs() << "\n=== Generated MLIR (before lowering) ===\n";
        mlirModule.print(llvm::errs());
        llvm::errs() << "\n";
    }

    // Register all passes before using PassManager
    mlir::registerAllPasses();

    // Lower MLIR to LLVM dialect
    PassManager pm(&ctx);
    // First, convert Arith operations to LLVM
    pm.addPass(mlir::createArithToLLVMConversionPass());
    // Then, convert Func and MemRef to LLVM
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    // Finally, reconcile any remaining casts
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (failed(pm.run(mlirModule))) {
        llvm::errs() << "Failed to lower MLIR to LLVM dialect\n";
        return;
    }

    if (debug_) {
        llvm::errs() << "\n=== MLIR after LLVM lowering ===\n";
        mlirModule.print(llvm::errs());
        llvm::errs() << "\n";
    }

    // Register LLVM dialect translations for execution engine
    mlir::DialectRegistry registry;
    mlir::registerAllToLLVMIRTranslations(registry);
    ctx.appendDialectRegistry(registry);

    // Create a JIT execution engine
    auto maybeEngine = ExecutionEngine::create(mlirModule);
    if (!maybeEngine) {
        llvm::errs() << "Failed to create execution engine\n";
        return;
    }

    auto& engine = maybeEngine.get();

    if (debug_) {
        llvm::errs() << "\n=== Executing main() ===\n";
    }

    // Execute main function
    if (auto error = engine->invokePacked("main")) {
        llvm::errs() << "Execution failed: " << error << "\n";
        return;
    }

    if (debug_) {
        llvm::errs() << "=== Execution complete ===\n";
    }
}

}  // namespace zenith::backend

