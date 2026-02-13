//===- ZenithTypes.h - Zenith dialect types --------------------*- C++ -*-===//
//
// This file defines the types for the Zenith dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZENITH_DIALECT_ZENITHTYPES_H
#define ZENITH_DIALECT_ZENITHTYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace zenith {

//===----------------------------------------------------------------------===//
// Zenith Type
//===----------------------------------------------------------------------===//

class ZenithType : public Type {
public:
    using Type::Type;

    static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// Integer Type
//===----------------------------------------------------------------------===//

class IntegerType : public Type::TypeBase<IntegerType, ZenithType,
                                           TypeStorage> {
public:
    using Base::Base;

    static IntegerType get(MLIRContext *context, unsigned width);
    unsigned getWidth() const;
};

//===----------------------------------------------------------------------===//
// Float Type
//===----------------------------------------------------------------------===//

class FloatType : public Type::TypeBase<FloatType, ZenithType, TypeStorage> {
public:
    using Base::Base;

    static FloatType get(MLIRContext *context);
};

//===----------------------------------------------------------------------===//
// String Type
//===----------------------------------------------------------------------===//

class StringType : public Type::TypeBase<StringType, ZenithType, TypeStorage> {
public:
    using Base::Base;

    static StringType get(MLIRContext *context);
};

} // namespace zenith
} // namespace mlir

#endif // ZENITH_DIALECT_ZENITHTYPES_H

