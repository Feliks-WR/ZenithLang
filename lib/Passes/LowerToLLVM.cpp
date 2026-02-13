//===- LowerToLLVM.cpp - Lower Zenith to LLVM dialect --------------------===//
//
// This file implements the pass to lower Zenith dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "Zenith/Passes/Passes.h"
#include "Zenith/Dialect/ZenithDialect.h"
#include "Zenith/Dialect/ZenithOps.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::zenith;

namespace {

//===----------------------------------------------------------------------===//
// Lowering Patterns
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpConversionPattern<ConstantOp> {
    using OpConversionPattern<ConstantOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, op.getValue());
        return success();
    }
};

struct AddOpLowering : public OpConversionPattern<AddOp> {
    using OpConversionPattern<AddOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(AddOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto resultType = op.getResult().getType();

        if (resultType.isIntOrIndex()) {
            rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
        } else if (resultType.isa<FloatType>()) {
            rewriter.replaceOpWithNewOp<LLVM::FAddOp>(op, adaptor.getLhs(),
                                                       adaptor.getRhs());
        } else {
            return failure();
        }

        return success();
    }
};

struct SubOpLowering : public OpConversionPattern<SubOp> {
    using OpConversionPattern<SubOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SubOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto resultType = op.getResult().getType();

        if (resultType.isIntOrIndex()) {
            rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
        } else if (resultType.isa<FloatType>()) {
            rewriter.replaceOpWithNewOp<LLVM::FSubOp>(op, adaptor.getLhs(),
                                                       adaptor.getRhs());
        } else {
            return failure();
        }

        return success();
    }
};

struct MulOpLowering : public OpConversionPattern<MulOp> {
    using OpConversionPattern<MulOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(MulOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto resultType = op.getResult().getType();

        if (resultType.isIntOrIndex()) {
            rewriter.replaceOpWithNewOp<LLVM::MulOp>(op, adaptor.getLhs(),
                                                      adaptor.getRhs());
        } else if (resultType.isa<FloatType>()) {
            rewriter.replaceOpWithNewOp<LLVM::FMulOp>(op, adaptor.getLhs(),
                                                       adaptor.getRhs());
        } else {
            return failure();
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LowerToLLVMPass
    : public PassWrapper<LowerToLLVMPass, OperationPass<ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLLVMPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() override {
        LLVMConversionTarget target(getContext());
        target.addLegalOp<ModuleOp>();

        LLVMTypeConverter typeConverter(&getContext());

        RewritePatternSet patterns(&getContext());
        patterns.add<ConstantOpLowering, AddOpLowering, SubOpLowering,
                     MulOpLowering>(typeConverter, &getContext());

        auto module = getOperation();
        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }

    StringRef getArgument() const final { return "zenith-lower-to-llvm"; }
    StringRef getDescription() const final {
        return "Lower Zenith dialect to LLVM dialect";
    }
};

} // namespace

std::unique_ptr<Pass> mlir::zenith::createLowerToLLVMPass() {
    return std::make_unique<LowerToLLVMPass>();
}

