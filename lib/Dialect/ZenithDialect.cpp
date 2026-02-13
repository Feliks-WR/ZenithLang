//===- ZenithDialect.cpp - Zenith dialect ---------------------------------===//
//
// This file implements the Zenith dialect.
//
//===----------------------------------------------------------------------===//

#include "Zenith/Dialect/ZenithDialect.h"
#include "Zenith/Dialect/ZenithOps.h"
#include "Zenith/Dialect/ZenithTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::zenith;

#include "Zenith/Dialect/ZenithOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Zenith dialect.
//===----------------------------------------------------------------------===//

void ZenithDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Zenith/Dialect/ZenithOps.cpp.inc"
        >();
    registerTypes();
}

void ZenithDialect::registerTypes() {
    // Register types here when implemented
}

void ZenithDialect::registerAttributes() {
    // Register attributes here when implemented
}

