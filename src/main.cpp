#include "antlr4-runtime.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "ZenithParserBaseVisitor.h"
#include "CodeGenerator.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

using namespace antlr4;
namespace fs = std::filesystem;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: zenith <file.zenith> [-o output] [--emit-c|--no-compile]\n";
        std::cerr << "  Default: compiles to executable with same name as input (no extension)\n";
        return 1;
    }

    // Parse command line
    std::string inputFile = argv[1];
    std::string outputFile;
    bool emitCOnly = false;
    bool noCompile = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            outputFile = argv[++i];
        } else if (arg == "--emit-c") {
            emitCOnly = true;
        } else if (arg == "--no-compile") {
            noCompile = true;
        }
    }

    // Read input file
    std::ifstream file(inputFile);
    if (!file) {
        std::cerr << "Error: cannot open file '" << inputFile << "'\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    file.close();

    // Parse
    ANTLRInputStream input(source);
    ZenithLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    ZenithParser parser(&tokens);

    // Get AST
    ZenithParser::ProgramContext *tree = parser.program();

    if (parser.getNumberOfSyntaxErrors() > 0) {
        std::cerr << "❌ Parse errors detected\n";
        return 1;
    }

    std::cout << "✓ Parsed successfully\n";

    // Generate C code
    CodeGenerator codegen;
    codegen.visit(tree);
    std::string generatedC = codegen.getGeneratedCode();

    // Determine output filename
    if (outputFile.empty()) {
        fs::path inPath(inputFile);
        outputFile = inPath.stem().string();  // Remove .zenith extension
    }

    // Write generated C code to file
    std::string cFile = outputFile + ".c";
    {
        std::ofstream cOut(cFile);
        cOut << generatedC;
        cOut.close();
    }
    std::cout << "✓ Generated C code: " << cFile << "\n";

    if (emitCOnly) {
        std::cout << generatedC;
        return 0;
    }

    if (noCompile) {
        return 0;
    }

    // Compile C code to executable using gcc
    std::string compileCmd = "gcc -o " + outputFile + " " + cFile;
    std::cout << "🔨 Compiling: " << compileCmd << "\n";
    int compileStatus = std::system(compileCmd.c_str());

    if (compileStatus != 0) {
        std::cerr << "❌ Compilation failed\n";
        return 1;
    }

    std::cout << "✓ Compiled successfully to: " << outputFile << "\n";
    std::cout << "Run with: ./" << outputFile << "\n";

    return 0;
}
