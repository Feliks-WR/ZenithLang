#ifndef CUSTOMLANG_DIALECT_H
#define CUSTOMLANG_DIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace customlang {

class CustomLangDialect : public Dialect {
  explicit CustomLangDialect(MLIRContext *context);

  void initialize();
  static StringRef getDialectNamespace() { return "customlang"; }
};

}  // namespace customlang
}  // namespace mlir

// Include generated dialect class
// #include "CustomLangDialect.h.inc"

#endif  // CUSTOMLANG_DIALECT_H
