#include "Zenith/ZenithDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>

using namespace mlir;
using namespace mlir::zenith;

namespace
{
    // Helper to get or create a global string in the module
    Value getOrCreateGlobalString(PatternRewriter &rewriter, Location loc, ModuleOp module,
                                        StringRef prefix, StringRef value) {
        auto *context = rewriter.getContext();
        
        // Create a unique name for the global
        std::string globalName = (prefix + "_" + std::to_string(llvm::hash_value(value))).str();

        LLVM::GlobalOp global;
        if (!((global = module.lookupSymbol<LLVM::GlobalOp>(globalName)))) {
            OpBuilder moduleBuilder(module.getContext());
            moduleBuilder.setInsertionPointToStart(module.getBody());

            // Create the null-terminated string attribute
            std::string nullTerminatedValue = value.str();
            nullTerminatedValue.push_back('\0');
            auto strAttr = rewriter.getStringAttr(nullTerminatedValue);
            
            // The type must be [length x i8] where length includes \0
            auto type = LLVM::LLVMArrayType::get(rewriter.getI8Type(), nullTerminatedValue.size());
        
            global = moduleBuilder.create<LLVM::GlobalOp>(
                loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
                globalName, strAttr, /*alignment=*/0);
        }

        return rewriter.create<LLVM::AddressOfOp>(loc,
            LLVM::LLVMPointerType::get(context), global.getSymName());
    }

    // Helper to get or create a global array in the module
    Value getOrCreateGlobalArray(PatternRewriter &rewriter, Location loc, ModuleOp module,
                                 StringRef prefix, DenseElementsAttr value) {
        auto *context = rewriter.getContext();
        const auto tensorType = cast<RankedTensorType>(value.getType());
        
        // Create a unique name for the global
        std::string globalName = (prefix + "_" + std::to_string(mlir::hash_value(value))).str();

        LLVM::GlobalOp global;
        if (!((global = module.lookupSymbol<LLVM::GlobalOp>(globalName)))) {
            OpBuilder moduleBuilder(module.getContext());
            moduleBuilder.setInsertionPointToStart(module.getBody());

            auto llvmElementType = rewriter.getI64Type();
            auto type = LLVM::LLVMArrayType::get(llvmElementType, tensorType.getNumElements());
        
            global = moduleBuilder.create<LLVM::GlobalOp>(
                loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
                globalName, value, /*alignment=*/0);
        }

        return rewriter.create<LLVM::AddressOfOp>(loc,
            LLVM::LLVMPointerType::get(context), global.getSymName());
    }

    // Convert zenith.constant to appropriate dialect operations
    struct constant_op_lowering final : OpRewritePattern<ConstantOp>
    {
        using OpRewritePattern<ConstantOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ConstantOp op,
                                      PatternRewriter &rewriter) const override {
            auto attr = op.getValue();
            const auto loc = op.getLoc();

            if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
                rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, intAttr);
                return success();
            }

            if (const auto strAttr = dyn_cast<StringAttr>(attr)) {
                // Handle string constants by creating a global string in the module
                auto module = op->getParentOfType<ModuleOp>();
                if (!module) {
                    return failure();
                }
                // Create the global string and get its address
                Value addressOf = getOrCreateGlobalString(rewriter, loc, module, "str_constant", strAttr.getValue());

                rewriter.replaceOp(op, {addressOf});
                return success();
            }

            if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
                // Handle array constants - replace with arith.constant for now
                // PrintOp will handle the conversion to global if needed
                rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, denseAttr);
                return success();
            }

            return failure();
        }
    };

    // Convert zenith.print to runtime function calls
    struct PrintOpLowering final : public OpRewritePattern<PrintOp> {
        using OpRewritePattern<PrintOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(PrintOp op,
                                      PatternRewriter &rewriter) const override {
            const auto loc = op.getLoc();
            const auto value = op.getValue();
            const auto type = value.getType();
            const auto module = op->getParentOfType<ModuleOp>();

            if (type.isInteger(64)) {
                auto funcType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {rewriter.getI64Type()});
                auto funcRef = getOrInsertFunc(rewriter, module, "zenith_print_i64", funcType);
                rewriter.create<LLVM::CallOp>(loc, funcType, funcRef, ValueRange{value});
            } else if (isa<LLVM::LLVMPointerType>(type)) {
                // Check if it's a string, or we should treat it as such
                auto funcType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {LLVM::LLVMPointerType::get(rewriter.getContext())});
                auto funcRef = getOrInsertFunc(rewriter, module, "zenith_print_str", funcType);
                rewriter.create<LLVM::CallOp>(loc, funcType, funcRef, ValueRange{value});
            } else if (const auto tensorType = dyn_cast<RankedTensorType>(type)) {
                // Array printing via runtime function
                
                // Get the data from the defining constant operation
                DenseElementsAttr attr;
                if (auto constOp = value.getDefiningOp<ConstantOp>()) {
                    attr = dyn_cast<DenseElementsAttr>(constOp.getValue());
                } else if (auto arithConstOp = value.getDefiningOp<arith::ConstantOp>()) {
                    attr = dyn_cast<DenseElementsAttr>(arithConstOp.getValue());
                }

                if (!attr) return failure();

                const Value ptr = getOrCreateGlobalArray(rewriter, loc, module, "arr_data", attr);
                const Value size = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 
                    rewriter.getI64IntegerAttr(tensorType.getDimSize(0)));

                auto funcType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(), 
                        {LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI64Type()});
                auto funcRef = getOrInsertFunc(rewriter, module, "zenith_print_array", funcType);
                
                rewriter.create<LLVM::CallOp>(loc, funcType, funcRef, ValueRange{ptr, size});
            } else {
                return failure();
            }

            rewriter.eraseOp(op);
            return success();
        }

    private:
        static FlatSymbolRefAttr getOrInsertFunc(PatternRewriter &rewriter, ModuleOp module, 
                                               StringRef name, LLVM::LLVMFunctionType type) {
            auto *context = rewriter.getContext();
            if (module.lookupSymbol<LLVM::LLVMFuncOp>(name))
                return SymbolRefAttr::get(context, name);

            OpBuilder moduleBuilder(context);
            moduleBuilder.setInsertionPointToStart(module.getBody());
            moduleBuilder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
            return SymbolRefAttr::get(context, name);
        }
    };

    struct ZenithToStandardPass final : public PassWrapper<ZenithToStandardPass, OperationPass<ModuleOp>> {
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZenithToStandardPass)

        void runOnOperation() override {
            const ModuleOp module = getOperation();
            ConversionTarget target(getContext());

            target.addLegalDialect<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect>();
            target.addIllegalDialect<ZenithDialect>();

            RewritePatternSet patterns(&getContext());
            patterns.add<constant_op_lowering, PrintOpLowering>(&getContext());

            if (failed(applyPartialConversion(module, target, std::move(patterns))))
                signalPassFailure();
        }
    };

    std::unique_ptr<mlir::Pass> createZenithToStandardPassImpl() {
        return std::make_unique<ZenithToStandardPass>();
    }
}// namespace

namespace mlir::zenith {
std::unique_ptr<mlir::Pass> createZenithToStandardPass() {
    return createZenithToStandardPassImpl();
}
}// namespace mlir::zenith
