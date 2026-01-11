#include "zenith/stdlib_registry.hpp"
#include <map>

namespace zenith::stdlib {
    class StdlibRegistry;

    // Embedded stdlib source code
    auto StdlibRegistry::CORE_MODULE = R"(
proc print(x: int)
proc print(x: float)
proc print(x: bool)
proc print(x: str)
proc print(x: [int])
proc print(x: [float])
proc print(x: [str])
)";

    auto StdlibRegistry::CONSTRAINTS_MODULE = R"(
// Constraint definitions for dependent types
// These are markers for the type system to validate constraints

constraint sorted
constraint reversed
constraint even
constraint odd
constraint positive
constraint negative
constraint non_negative
constraint non_positive
constraint not_nan
constraint not_inf
constraint finite
constraint non_empty
constraint ascii
)";

auto StdlibRegistry::ARRAY_MODULE = R"(
pure len(arr: [int]) -> int
pure len(arr: [float]) -> int
pure len(arr: [str]) -> int
pure len(arr: [bool]) -> int

pure index(arr: [int], idx: int) -> int
pure index(arr: [float], idx: int) -> float
pure index(arr: [str], idx: int) -> str

pure slice(arr: [int], start: int, end: int) -> [int]
pure slice(arr: [float], start: int, end: int) -> [float]

pure is_sorted(arr: [int]) -> bool
pure is_sorted(arr: [float]) -> bool

pure sort(arr: [int]) -> [int]
pure sort(arr: [float]) -> [float]

pure reverse(arr: [int]) -> [int]
pure reverse(arr: [float]) -> [float]
)";

StdlibRegistry& StdlibRegistry::instance() {
    static StdlibRegistry instance;
    return instance;
}

void StdlibRegistry::loadStdlib() {
    if (stdlib_loaded) return;
    // In a real implementation, we would parse these modules
    // and register their function/constraint definitions
    stdlib_loaded = true;
}

std::string StdlibRegistry::getModuleSource(const std::string& moduleName) {
    if (moduleName == "core") return CORE_MODULE;
    if (moduleName == "constraints") return CONSTRAINTS_MODULE;
    if (moduleName == "array") return ARRAY_MODULE;
    return "";
}

bool StdlibRegistry::isConstraintDefined(const std::string& constraintName) {
    static const std::vector<std::string> defined_constraints = {
        "sorted", "reversed", "even", "odd", "positive", "negative",
        "non_negative", "non_positive", "not_nan", "not_inf", "finite",
        "non_empty", "ascii"
    };

    for (const auto& c : defined_constraints) {
        if (c == constraintName) return true;
    }
    return false;
}

}  // namespace zenith::stdlib

