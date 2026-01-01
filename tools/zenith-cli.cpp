#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "antlr4-runtime.h"
#include "zenith/frontend/parser.hpp"
#include "zenith/backend/pipeline.hpp"
#include "zenith/backend/mlir_pass.hpp"
#include "zenith/backend/execution_pass.hpp"

enum class OutputMode {
    EXECUTE,  // Default: execute the code
    MLIR,     // Output MLIR IR
};

struct Config {
    std::string inputFile;
    OutputMode mode = OutputMode::EXECUTE;
    bool debug = false;
};

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " <input_file> [options]\n";
    std::cerr << "\nOptions:\n";
    std::cerr << "  --mlir, -m     Output MLIR IR instead of executing\n";
    std::cerr << "  --debug, -d    Enable debug output\n";
    std::cerr << "  --help, -h     Show this help message\n";
    std::cerr << "\nDefault behavior: Execute the code\n";
}

Config parseArgs(const int argc, const char* argv[]) {
    if (argc < 2) {
        throw std::invalid_argument("No input file specified");
    }

    Config config;
    config.inputFile = argv[1];

    // Parse flags
    for (int i = 2; i < argc; ++i) {
        if (std::string arg = argv[i]; arg == "--mlir" || arg == "-m") {
            config.mode = OutputMode::MLIR;
        } else if (arg == "--debug" || arg == "-d") {
            config.debug = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    return config;
}

int main(const int argc, const char* argv[]) {
    try {
        // Parse command-line arguments
        auto [inputFile, mode, debug] = parseArgs(argc, argv);

        // Parse source to HIR
        const auto hir = zenith::frontend::Parser::parseFile(inputFile);
        if (!hir) {
            std::cerr << "Failed to parse file\n";
            return 1;
        }

        // Build and run pipeline based on output mode
        zenith::backend::Pipeline pipeline;

        switch (mode) {
            case OutputMode::EXECUTE: {
                // Execute via MLIR -> LLVM lowering and JIT
                zenith::backend::ExecutionPassFactory execFactory(debug);
                pipeline.addPass(execFactory);
                break;
            }
            case OutputMode::MLIR: {
                // Just emit MLIR
                zenith::backend::MLIRPassFactory mlirFactory(std::cout, debug);
                pipeline.addPass(mlirFactory);
                break;
            }
        }

        // Run the pipeline
        pipeline.run(*hir);

        return 0;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        printUsage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
