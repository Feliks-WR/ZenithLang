#include "Dialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::customlang;

void CustomLangDialect::initialize() {
  // Register operations and types here
}

CustomLangDialect::CustomLangDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<CustomLangDialect>()) {
  initialize();
}
