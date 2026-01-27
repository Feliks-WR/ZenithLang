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

# Testing with Google Test
enable_testing()
find_package(GTest)

if(GTest_FOUND)
    # Parser unit tests
    add_executable(test_parser
        tests/test_parser.cpp
        ${ANTLR_GEN_SRC}
    )
    target_link_libraries(test_parser
        PRIVATE
        GTest::gtest
        GTest::gtest_main
        antlr4-runtime
    )
    target_include_directories(test_parser PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${ANTLR_GEN_DIR}"
    )
    add_test(NAME ParserTests COMMAND test_parser)
    
    # Code generator unit tests
    add_executable(test_codegen
        tests/test_codegen.cpp
        src/CodeGenerator.cpp
        ${ANTLR_GEN_SRC}
    )
    target_link_libraries(test_codegen
        PRIVATE
        GTest::gtest
        GTest::gtest_main
        antlr4-runtime
    )
    target_include_directories(test_codegen PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${ANTLR_GEN_DIR}"
    )
    add_test(NAME CodeGenTests COMMAND test_codegen)
    
    # Integration tests
    add_executable(integration_test
        tests/integration_test.cpp
    )
    target_link_libraries(integration_test
        PRIVATE
        GTest::gtest
        GTest::gtest_main
    )
    add_test(NAME IntegrationTests COMMAND integration_test)
    
    message(STATUS "✓ Google Test framework found - unit tests enabled")
else()
    message(WARNING "⚠ Google Test not found - tests disabled")
    message(STATUS "  Install with: sudo apt-get install libgtest-dev")
endif()

# Add a simple end-to-end test that compiles a sample file
add_test(NAME compile_basic COMMAND zenith ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basic.zenith -o test_output)
set_tests_properties(compile_basic PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

message(STATUS "✓ Zenith compiler built successfully")
message(STATUS "  Run: ./zenith <file.zenith>")
message(STATUS "  Test: ctest --output-on-failure")
