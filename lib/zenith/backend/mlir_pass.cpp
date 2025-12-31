#include "zenith/backend/mlir_pass.hpp"
#include "zenith/ir/hir.hpp"
#include <functional>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>


namespace zenith::ir
{
    struct HirModule;
}

namespace zenith::backend {

void MLIRPass::run(const ir::HirModule& hir) {
    using namespace mlir;

    // Create MLIR context and load dialects
    MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();

    OpBuilder builder(&ctx);
    auto mlirModule = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(mlirModule.getBody());

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
                return b.create<mlir::arith::ConstantOp>(
                    b.getUnknownLoc(), ty, b.getBoolAttr(e.boolVal));
            }
            case ir::HirExpr::Kind::StringLit:
            case ir::HirExpr::Kind::Id:
            case ir::HirExpr::Kind::Array:
            case ir::HirExpr::Kind::Call:
                return nullptr;
            case ir::HirExpr::Kind::Group:
            case ir::HirExpr::Kind::Unsafe:
                if (!e.elements.empty()) return lowerExpr(e.elements[0], b, blk);
                return nullptr;
        }
        return nullptr;
    };

    // Lambda: lower statement
    std::function<void(const ir::HirStmt&, OpBuilder&, mlir::Block*)> lowerStmt;
    lowerStmt = [&](const ir::HirStmt& stmt, OpBuilder& b, mlir::Block* blk) {
        b.setInsertionPointToEnd(blk);
        switch (stmt.kind) {
            case ir::HirStmt::Kind::Assign: {
                auto rhsVal = lowerExpr(stmt.expr, b, blk);
                break;
            }
            case ir::HirStmt::Kind::Call: {
                break;
            }
        }
    };

    // Lower top-level statements
    for (const auto& stmt : hir.topLevel) {
        lowerStmt(stmt, builder, mlirModule.getBody());
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

    // Print MLIR module to stdout
    llvm::outs() << mlirModule;
}

}  // namespace zenith::backend

