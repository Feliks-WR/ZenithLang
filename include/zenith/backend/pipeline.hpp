#ifndef ZENITH_BACKEND_PIPELINE_HPP
#define ZENITH_BACKEND_PIPELINE_HPP

#include "pass.hpp"
#include "zenith/ir/hir.hpp"
#include <memory>
#include <vector>

namespace zenith::backend {

/// \brief Compiler pass pipeline (Builder pattern + Chain of Responsibility).
class Pipeline final {
public:
    Pipeline() = default;

    Pipeline& addPass(std::unique_ptr<Pass> pass) {
        passes_.push_back(std::move(pass));
        return *this;
    }

    Pipeline& addPass(PassFactory& factory) {
        passes_.push_back(factory.create());
        return *this;
    }

    void run(const ir::HirModule& hir) const
    {
        for (auto& pass : passes_) {
            if (pass) {
                pass->run(hir);
            }
        }
    }

    void clear() {
        passes_.clear();
    }

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};

}  // namespace zenith::backend

#endif  // ZENITH_BACKEND_PIPELINE_HPP

