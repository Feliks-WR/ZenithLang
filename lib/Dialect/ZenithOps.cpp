//===- ZenithOps.cpp - Zenith dialect ops ---------------------------------===//
//
// This file implements the operations for the Zenith dialect.
//
//===----------------------------------------------------------------------===//

#include "Zenith/Dialect/ZenithOps.h"
#include "Zenith/Dialect/ZenithDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::zenith;

#define GET_OP_CLASSES
#include "Zenith/Dialect/ZenithOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
    return getValue();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs)
        return {};

    // Fold integer addition
    if (auto lhsInt = llvm::dyn_cast<IntegerAttr>(lhs)) {
        if (auto rhsInt = llvm::dyn_cast<IntegerAttr>(rhs)) {
            return IntegerAttr::get(lhsInt.getType(),
                                   lhsInt.getValue() + rhsInt.getValue());
        }
    }

    // Fold float addition
    if (auto lhsFp = llvm::dyn_cast<FloatAttr>(lhs)) {
        if (auto rhsFp = llvm::dyn_cast<FloatAttr>(rhs)) {
            return FloatAttr::get(lhsFp.getType(),
                                 lhsFp.getValue() + rhsFp.getValue());
        }
    }

    return {};
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs)
        return {};

    if (auto lhsInt = llvm::dyn_cast<IntegerAttr>(lhs)) {
        if (auto rhsInt = llvm::dyn_cast<IntegerAttr>(rhs)) {
            return IntegerAttr::get(lhsInt.getType(),
                                   lhsInt.getValue() - rhsInt.getValue());
        }
    }

    if (auto lhsFp = llvm::dyn_cast<FloatAttr>(lhs)) {
        if (auto rhsFp = llvm::dyn_cast<FloatAttr>(rhs)) {
            return FloatAttr::get(lhsFp.getType(),
                                 lhsFp.getValue() - rhsFp.getValue());
        }
    }

    return {};
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs)
        return {};

    if (auto lhsInt = llvm::dyn_cast<IntegerAttr>(lhs)) {
        if (auto rhsInt = llvm::dyn_cast<IntegerAttr>(rhs)) {
            return IntegerAttr::get(lhsInt.getType(),
                                   lhsInt.getValue() * rhsInt.getValue());
        }
    }

    if (auto lhsFp = llvm::dyn_cast<FloatAttr>(lhs)) {
        if (auto rhsFp = llvm::dyn_cast<FloatAttr>(rhs)) {
            return FloatAttr::get(lhsFp.getType(),
                                 lhsFp.getValue() * rhsFp.getValue());
        }
    }

    return {};
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs) {
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
    state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
    auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                           ArrayRef<Type> results,
                           function_interface_impl::VariadicFlag,
                           std::string &) {
        return builder.getFunctionType(argTypes, results);
    };

    return function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
    function_interface_impl::printFunctionOp(
        p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
}

