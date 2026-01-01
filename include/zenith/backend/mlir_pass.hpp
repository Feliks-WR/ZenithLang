#ifndef ZENITH_BACKEND_MLIR_PASS_HPP
#define ZENITH_BACKEND_MLIR_PASS_HPP

#include "pass.hpp"
#include <memory>
#include <iostream>

namespace zenith::backend {

/// \brief MLIR generation pass (Concrete Strategy).
class MLIRPass final : public Pass {
public:
    explicit MLIRPass(std::ostream& out = std::cout, bool debug = false)
        : output_(out), debug_(debug) {}
    void run(const ir::HirModule& hir) override;

private:
    std::ostream& output_;
    bool debug_;
};

/// \brief Factory for MLIR pass.
class MLIRPassFactory final : public PassFactory {
public:
    explicit MLIRPassFactory(std::ostream& out = std::cout, bool debug = false)
        : output_(out), debug_(debug) {}
    std::unique_ptr<Pass> create() override {
        return std::make_unique<MLIRPass>(output_, debug_);
    }

private:
    std::ostream& output_;
    bool debug_;
};

}  // namespace zenith::backend

#endif  // ZENITH_BACKEND_MLIR_PASS_HPP

