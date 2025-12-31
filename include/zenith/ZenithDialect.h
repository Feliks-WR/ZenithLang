#ifndef ZENITH_ZENITHDIALECT_H
#define ZENITH_ZENITHDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

// Include generated headers
#include "Zenith/ZenithOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Zenith/ZenithOps.h.inc"

#endif // ZENITH_ZENITHDIALECT_H
