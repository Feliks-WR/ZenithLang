cmake_minimum_required(VERSION 3.20)
project(ZenithLanguage)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ANTLR4 - generate at configure time
set(ANTLR_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/antlr_gen")
file(MAKE_DIRECTORY "${ANTLR_GEN_DIR}")

execute_process(
    COMMAND antlr4_new -Dlanguage=Cpp -visitor -no-listener 
            -o "${ANTLR_GEN_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/grammar/ZenithLexer.g4"
    COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
    COMMAND antlr4_new -Dlanguage=Cpp -visitor -no-listener
            -o "${ANTLR_GEN_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/grammar/ZenithParser.g4"
    COMMAND_ERROR_IS_FATAL ANY
)

# Main compiler executable (no MLIR yet, just parsing)
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${ANTLR_GEN_DIR}"
    "/usr/local/include"
)

# Collect generated ANTLR files
file(GLOB ANTLR_GEN_SRC "${ANTLR_GEN_DIR}/*.cpp")

add_executable(zenith
    src/main.cpp
    src/CodeGenerator.cpp
    ${ANTLR_GEN_SRC}
)

target_link_libraries(zenith
    PRIVATE
    antlr4-runtime
)

# MLIR / LLVM integration (optional, for future emit-llvm / jit modes)
find_package(LLVM CONFIG)
find_package(MLIR CONFIG)

if(LLVM_FOUND AND MLIR_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    include(AddMLIR)
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
    add_definitions(-DUSE_MLIR)
    
    llvm_map_components_to_libnames(REQ_LLVM_LIBS support core irreader native)
    target_link_libraries(zenith
        PRIVATE
        ${REQ_LLVM_LIBS}
        MLIRExecutionEngine
        MLIRTransforms
        MLIRParser
        MLIRIR
        MLIRSupport
    )
    message(STATUS "✓ MLIR support enabled")
else()
    message(STATUS "⚠ MLIR not found; building parser-only mode")
endif()

enable_testing()

# Add a simple parsing test that runs the `zenith` binary on a sample file
add_test(NAME parse_basic COMMAND zenith ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basic.zenith)

message(STATUS "✓ Zenith compiler built successfully")
message(STATUS "  Run: ./zenith <file.zenith>")
