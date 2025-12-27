#ifndef ZENITH_PASSES_H
#define ZENITH_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace zenith {

/// Create a pass that lowers Zenith dialect to standard dialects
std::unique_ptr<Pass> createZenithToStandardPass();

} // namespace zenith
} // namespace mlir

#endif // ZENITH_PASSES_H
