#pragma once

#ifndef ZENITH_BACKEND_EXECUTION_PASS_HPP
#define ZENITH_BACKEND_EXECUTION_PASS_HPP

#include "pass.hpp"
#include <memory>

namespace zenith::backend {

/// \brief Execution pass - compiles and runs MLIR code via LLVM JIT.
class ExecutionPass final : public Pass {
public:
    explicit ExecutionPass(bool debug = false) : debug_(debug) {}
    void run(const ir::HirModule& hir) override;

private:
    bool debug_;
};

/// \brief Factory for execution pass.
class ExecutionPassFactory final : public PassFactory {
public:
    explicit ExecutionPassFactory(bool debug = false) : debug_(debug) {}
    std::unique_ptr<Pass> create() override {
        return std::make_unique<ExecutionPass>(debug_);
    }

private:
    bool debug_;
};

}  // namespace zenith::backend

#endif  // ZENITH_BACKEND_EXECUTION_PASS_HPP

