//===- ZenithOps.h - Zenith dialect ops -------------------------*- C++ -*-===//
//
// This file defines the operations for the Zenith dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZENITH_DIALECT_ZENITHOPS_H
#define ZENITH_DIALECT_ZENITHOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Zenith/Dialect/ZenithOps.h.inc"

#endif // ZENITH_DIALECT_ZENITHOPS_H

