#include "zenith/backend/mlir_pass.hpp"
#include "zenith/ir/hir.hpp"
#include <functional>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"


namespace zenith::ir
{
    struct HirModule;
}

namespace zenith::backend {

void MLIRPass::run(const ir::HirModule& hir) {
    using namespace mlir;

    // Create MLIR context and load dialects
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect>();
    ctx.loadDialect<arith::ArithDialect>();
    ctx.loadDialect<memref::MemRefDialect>();

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
                return b.create<mlir::arith::ConstantOp>(
                    b.getUnknownLoc(), ty, b.getI64IntegerAttr(e.intVal));
            }
            case ir::HirExpr::Kind::FloatLit: {
                auto ty = b.getF64Type();
                return b.create<mlir::arith::ConstantOp>(
                    b.getUnknownLoc(), ty, b.getF64FloatAttr(e.floatVal));
            }
            case ir::HirExpr::Kind::BoolLit: {
                auto ty = b.getI1Type();
                return b.create<arith::ConstantOp>(
                    b.getUnknownLoc(), ty, b.getBoolAttr(e.boolVal));
            }
            case ir::HirExpr::Kind::StringLit: {
                // Create an immutable global string (Rust-style: length + UTF-8 bytes, no \0)
                // String layout in memory: struct { length: i64, data: [i8 x N] }

                std::string globalName = "zenith_str_" + std::to_string(stringCounter++);
                const size_t strLen = e.strVal.length();

                // Create the byte array for string content
                SmallVector<int8_t> strBytes;
                for (const char c : e.strVal) {
                    strBytes.push_back(static_cast<int8_t>(c));
                }

                // Save current insertion point
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPointToStart(mlirModule.getBody());

                // Create memref type for string data: memref<Nxi8>
                const auto i8Type = b.getI8Type();
                const auto memrefType = MemRefType::get({static_cast<int64_t>(strLen)}, i8Type);

                // Create dense elements attribute for initial value
                const auto tensorType = RankedTensorType::get({static_cast<int64_t>(strLen)}, i8Type);
                auto initialValue = DenseElementsAttr::get(tensorType, llvm::ArrayRef(strBytes));

                // Create memref.global with a constant attribute
                auto globalOp = b.create<memref::GlobalOp>(
                    b.getUnknownLoc(),
                    globalName,
                    b.getStringAttr("private"),
                    memrefType,
                    initialValue,
                    /*constant=*/true,
                    /*alignment=*/nullptr
                );

                // Add length metadata as an attribute (for Rust-style string descriptor)
                globalOp->setAttr("zenith.string.length", b.getI64IntegerAttr(strLen));
                globalOp->setAttr("zenith.string.encoding", b.getStringAttr("utf8"));

                // Return nullptr for now (in real usage, would return memref reference)
                return nullptr;
            }
            case ir::HirExpr::Kind::Array: {
                // Lower array elements to constants
                SmallVector<Value> elemVals;
                Type elemType = nullptr;

                for (const auto& elem : e.elements) {
                    if (const auto val = lowerExpr(elem, b, blk)) {
                        elemVals.push_back(val);
                        if (!elemType) elemType = val.getType();
                    }
                }

                if (elemVals.empty()) return nullptr;

                // Create a memref to hold the array: memref<Nxtype>
                auto arraySize = static_cast<int64_t>(elemVals.size());
                auto memrefType = MemRefType::get({arraySize}, elemType);

                // Allocate stack space for the array
                auto allocaOp = b.create<memref::AllocaOp>(
                    b.getUnknownLoc(),
                    memrefType
                );

                // Store each element into the memref
                for (size_t i = 0; i < elemVals.size(); ++i) {
                    auto indexVal = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), i);
                    b.create<memref::StoreOp>(
                        b.getUnknownLoc(),
                        elemVals[i],
                        allocaOp,
                        ValueRange{indexVal}
                    );
                }

                // Return the memref (represents the array)
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
                        if (lhsVal.getType().isInteger(64)) {
                            return b.create<arith::SubIOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        } else {
                            return b.create<arith::SubFOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        }
                    }
                    case ir::HirExpr::BinOpKind::Mul: {
                        if (lhsVal.getType().isInteger(64)) {
                            return b.create<arith::MulIOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        } else {
                            return b.create<arith::MulFOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        }
                    }
                    case ir::HirExpr::BinOpKind::Div: {
                        if (lhsVal.getType().isInteger(64)) {
                            return b.create<arith::DivSIOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        } else {
                            return b.create<arith::DivFOp>(b.getUnknownLoc(), lhsVal, rhsVal);
                        }
                    }
                    case ir::HirExpr::BinOpKind::Concat: {
                        // String/array concatenation would require additional implementation
                        return nullptr;
                    }
                }
                return nullptr;
            }
        }
        return nullptr;
    };

    // Lambda: lower statement
    const std::function lowerStmt = [&](const ir::HirStmt& stmt, OpBuilder& b, mlir::Block* blk){
        b.setInsertionPointToEnd(blk);

        // Debug output to stderr (only if a debug_ flag is set)
        if (debug_) {
            llvm::errs() << "Lowering assignment: " << stmt.name << " = ";
        }

        switch (stmt.kind)
        {
        case ir::HirStmt::Kind::Assign:
            {
                // Show what type of value is being assigned (only in debug mode)
                if (debug_) {
                    switch (stmt.expr.kind) {
                        case ir::HirExpr::Kind::IntLit:
                            llvm::errs() << stmt.expr.intVal << " (int)\n";
                            break;
                        case ir::HirExpr::Kind::StringLit:
                            llvm::errs() << "\"" << stmt.expr.strVal << "\" (string - UTF-8 immutable)\n";
                            break;
                        case ir::HirExpr::Kind::Array:
                            llvm::errs() << "[array with " << stmt.expr.elements.size() << " elements]\n";
                            break;
                        default:
                            llvm::errs() << "(other type)\n";
                    }
                }

                auto rhsVal = lowerExpr(stmt.expr, b, blk);
                break;
            }
        case ir::HirStmt::Kind::Call:
            {
                if (debug_) {
                    llvm::errs() << "call: " << stmt.name << "()\n";
                }
                break;
            }
        }
    };

    // Lower top-level statements
    if (debug_) {
        llvm::errs() << "=== Processing Zenith File ===\n";
    }
    for (const auto& stmt : hir.topLevel) {
        lowerStmt(stmt, builder, mlirModule.getBody());
    }
    if (debug_) {
        llvm::errs() << "=== MLIR Output ===\n";
        llvm::errs().flush();
    }

    // Lower function declarations
    for (const auto& fn : hir.functions) {
        Type retTy = builder.getNoneType();

        SmallVector<Type> argTys;
        for (const auto& _ : fn.params) {
            argTys.push_back(builder.getI64Type());
        }
        auto fnTy = builder.getFunctionType(argTys, retTy);

        auto funcOp = builder.create<mlir::func::FuncOp>(
            builder.getUnknownLoc(), fn.name, fnTy);
        auto& body = funcOp.getBody();
        mlir::Block* bodyBlk = &body.emplaceBlock();

        if (fn.exprBody) {
            if (auto val = lowerExpr(*fn.exprBody, builder, bodyBlk)) {
                builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), val);
            }
        } else {
            for (const auto& stmt : fn.body) {
                lowerStmt(stmt, builder, bodyBlk);
            }
            builder.setInsertionPointToEnd(bodyBlk);
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        }
    }

    // Print MLIR module to stdout with pretty formatting
    mlirModule.print(llvm::outs());
    llvm::outs() << "\n";
}

}  // namespace zenith::backend

