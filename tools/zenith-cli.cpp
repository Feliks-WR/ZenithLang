#include <iostream>
#include <string>
#include <stdexcept>
#include <memory>
#include "zenith/frontend/parser.hpp"
#include "zenith/backend/pipeline.hpp"
#include "zenith/backend/mlir_pass.hpp"

enum class OutputMode {
    MLIR,
};

OutputMode parseOutputMode(const std::string& modeStr) {
    if (modeStr == "mlir") return OutputMode::MLIR;
    throw std::invalid_argument("Unknown output mode: " + modeStr);
}

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " <input_file> [output_mode]\n";
    std::cerr << "Output modes:\n";
    std::cerr << "  mlir   - Output MLIR IR (default)\n";
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFile = argv[1];
    OutputMode mode = OutputMode::MLIR;

    if (argc >= 3) {
        try {
            mode = parseOutputMode(argv[2]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: " << e.what() << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    try {
        // Parse source to HIR
        auto hir = zenith::frontend::Parser::parseFile(inputFile);
        if (!hir) {
            std::cerr << "Failed to parse file\n";
            return 1;
        }

        // Build and run pipeline based on output mode
        zenith::backend::Pipeline pipeline;

        switch (mode) {
            case OutputMode::MLIR: {
                zenith::backend::MLIRPassFactory mlirFactory(std::cout);
                pipeline.addPass(mlirFactory);
                break;
            }
        }

        pipeline.run(*hir);
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

