#ifndef ZENITH_BACKEND_PASS_HPP
#define ZENITH_BACKEND_PASS_HPP

#include "zenith/ir/hir.hpp"
#include <memory>

namespace zenith::backend {

/// \brief Base interface for compiler passes (Strategy pattern).
class Pass {
public:
    virtual ~Pass() = default;
    virtual void run(const ir::HirModule& hir) = 0;
};

/// \brief Factory for creating passes (Abstract Factory pattern).
class PassFactory {
public:
    virtual ~PassFactory() = default;
    virtual std::unique_ptr<Pass> create() = 0;
};

}  // namespace zenith::backend

#endif  // ZENITH_BACKEND_PASS_HPP

