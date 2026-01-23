#ifndef CUSTOMLANG_OPS_H
#define CUSTOMLANG_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

using namespace mlir;

namespace mlir {
namespace customlang {

// Forward declarations
class CustomLangDialect;

}  // namespace customlang
}  // namespace mlir

// Include custom op definitions
#define GET_OP_CLASSES
#include "CustomLang.h.inc"

#endif  // CUSTOMLANG_OPS_H
