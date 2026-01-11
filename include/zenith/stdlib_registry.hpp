#ifndef ZENITH_STDLIB_REGISTRY_HPP
#define ZENITH_STDLIB_REGISTRY_HPP

#include <string>
#include <vector>
#include <memory>

namespace zenith::stdlib {

// Registry of standard library modules
class StdlibRegistry {
public:
    static StdlibRegistry& instance();

    // Load all stdlib modules at startup
    void loadStdlib();

    // Get stdlib module source by name
    std::string getModuleSource(const std::string& moduleName);

    // Check if a constraint is defined in stdlib
    bool isConstraintDefined(const std::string& constraintName);

private:
    StdlibRegistry() = default;
    bool stdlib_loaded = false;

    // Embedded stdlib sources
    static const char* CORE_MODULE;
    static const char* CONSTRAINTS_MODULE;
    static const char* ARRAY_MODULE;
};

}  // namespace zenith::stdlib

#endif  // ZENITH_STDLIB_REGISTRY_HPP

