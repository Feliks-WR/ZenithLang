#ifndef ZENITH_BACKEND_MLIR_PASS_HPP
#define ZENITH_BACKEND_MLIR_PASS_HPP

#include "zenith/backend/pass.hpp"
#include <memory>
#include <iostream>

#include "pass.hpp"

namespace zenith::backend {

/// \brief MLIR generation pass (Concrete Strategy).
class MLIRPass : public Pass {
public:
    explicit MLIRPass(std::ostream& out = std::cout) : output_(out) {}
    void run(const ir::HirModule& hir) override;

private:
    std::ostream& output_;
};

/// \brief Factory for MLIR pass.
class MLIRPassFactory : public PassFactory {
public:
    explicit MLIRPassFactory(std::ostream& out = std::cout) : output_(out) {}
    std::unique_ptr<Pass> create() override {
        return std::make_unique<MLIRPass>(output_);
    }

private:
    std::ostream& output_;
};

}  // namespace zenith::backend

#endif  // ZENITH_BACKEND_MLIR_PASS_HPP

