#ifndef ZENITH_FRONTEND_PARSER_HPP
#define ZENITH_FRONTEND_PARSER_HPP

#include "zenith/ir/hir.hpp"
#include <memory>
#include <string>

namespace zenith::frontend {

/// \brief Parses Zenith source code into a HIR module.
class Parser {
public:
    static std::unique_ptr<ir::HirModule> parseFile(const std::string& filename);
    static std::unique_ptr<ir::HirModule> parseString(const std::string& source);
};

}  // namespace zenith::frontend

#endif  // ZENITH_FRONTEND_PARSER_HPP

