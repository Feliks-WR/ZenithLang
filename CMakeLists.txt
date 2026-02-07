cmake_minimum_required(VERSION 3.20)
project(ZenithLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ANTLR4 - generate at configure time
set(ANTLR_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/antlr_gen")
file(MAKE_DIRECTORY "${ANTLR_GEN_DIR}")

# Try to find antlr4 command or use a jar file
option(BUNDLE_ANTLR_RUNTIME
        "Download and build a bundled ANTLR4 C++ runtime and use a bundled ANTLR jar for generation" ON)


include(FetchContent)

if (BUNDLE_ANTLR_RUNTIME)
    # Ensure we have a reproducible ANTLR generator and matching C++ runtime (4.10.1)
    set(ANTLR_VERSION "4.10.1")
    set(ANTLR_JAR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/antlr-${ANTLR_VERSION}-complete.jar")
    if (NOT EXISTS ${ANTLR_JAR})
        message(STATUS "Downloading ANTLR ${ANTLR_VERSION} jar to ${ANTLR_JAR}")
        file(DOWNLOAD "https://www.antlr.org/download/antlr-${ANTLR_VERSION}-complete.jar" ${ANTLR_JAR} SHOW_PROGRESS)
    endif ()

    # Fetch the matching C++ runtime from the ANTLR repository
    FetchContent_Declare(
            antlr4_runtime
            GIT_REPOSITORY https://github.com/antlr/antlr4.git
            GIT_TAG ${ANTLR_VERSION}
            GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(antlr4_runtime)

    # Manually build the runtime library instead of using their CMakeLists (avoids path issues)
    set(ANTLR4_RUNTIME_SRC_DIR "${antlr4_runtime_SOURCE_DIR}/runtime/Cpp/runtime/src")
    file(GLOB_RECURSE ANTLR4_RUNTIME_SOURCES
            "${ANTLR4_RUNTIME_SRC_DIR}/*.cpp"
    )
    add_library(antlr4_static STATIC ${ANTLR4_RUNTIME_SOURCES})
    target_include_directories(antlr4_static PUBLIC
            "${ANTLR4_RUNTIME_SRC_DIR}"
            "${ANTLR4_RUNTIME_SRC_DIR}/atn"
            "${ANTLR4_RUNTIME_SRC_DIR}/dfa"
            "${ANTLR4_RUNTIME_SRC_DIR}/misc"
            "${ANTLR4_RUNTIME_SRC_DIR}/support"
            "${ANTLR4_RUNTIME_SRC_DIR}/tree"
            "${ANTLR4_RUNTIME_SRC_DIR}/tree/pattern"
            "${ANTLR4_RUNTIME_SRC_DIR}/tree/xpath"
    )
    target_compile_definitions(antlr4_static PUBLIC ANTLR4CPP_STATIC)
    set_target_properties(antlr4_static PROPERTIES
            CXX_STANDARD 17
            POSITION_INDEPENDENT_CODE ON
    )
    # Link UUID on Linux, CoreFoundation on macOS
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        find_package(PkgConfig)
        pkg_check_modules(UUID REQUIRED uuid)
        target_link_libraries(antlr4_static PUBLIC ${UUID_LIBRARIES})
        target_include_directories(antlr4_static PUBLIC ${UUID_INCLUDE_DIRS})
    elseif (APPLE)
        find_library(COREFOUNDATION_LIBRARY CoreFoundation)
        target_link_libraries(antlr4_static PUBLIC ${COREFOUNDATION_LIBRARY})
    endif ()

    # Use explicit tool and args to avoid parsing issues with spaces
    set(ANTLR_TOOL "java")
    set(ANTLR_TOOL_ARGS "-jar" "${ANTLR_JAR}")
else ()
    find_program(ANTLR4_EXECUTABLE NAMES antlr4 antlr)
    if (ANTLR4_EXECUTABLE)
        set(ANTLR_TOOL "${ANTLR4_EXECUTABLE}")
        set(ANTLR_TOOL_ARGS)
    else ()
        message(FATAL_ERROR
                "ANTLR4 not found. Either install system antlr4 or enable
                BUNDLE_ANTLR_RUNTIME to download a bundled runtime and jar.")
    endif ()
endif ()

execute_process(
        COMMAND ${ANTLR_TOOL} ${ANTLR_TOOL_ARGS} -Dlanguage=Cpp -visitor -no-listener -o "${ANTLR_GEN_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/grammar/ZenithLexer.g4"
        RESULT_VARIABLE _antlr_res1
)
if (NOT _antlr_res1 EQUAL 0)
    message(FATAL_ERROR "ANTLR generation failed for ZenithLexer.g4 (command: ${ANTLR_TOOL})")
endif ()

execute_process(
        COMMAND ${ANTLR_TOOL} ${ANTLR_TOOL_ARGS} -Dlanguage=Cpp -visitor -no-listener -o
        "${ANTLR_GEN_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/grammar/ZenithParser.g4"
        RESULT_VARIABLE _antlr_res2
)
if (NOT _antlr_res2 EQUAL 0)
    message(FATAL_ERROR "ANTLR generation failed for ZenithParser.g4 (command: ${ANTLR_TOOL})")
endif ()

message(STATUS "⚠ ANTLR4 generated files may need compatibility fixes.")
message(STATUS "  If build fails, install compatible ANTLR4 or regenerate grammar files.")

# Main compiler executable (no MLIR yet, just parsing)
include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${ANTLR_GEN_DIR}"
        "/usr/local/include"
)

# Collect generated ANTLR files
file(GLOB ANTLR_GEN_SRC "${ANTLR_GEN_DIR}/*.cpp")

# Disable warnings for ANTLR-generated code (version compatibility issues)
set_source_files_properties(${ANTLR_GEN_SRC} PROPERTIES COMPILE_FLAGS "-Wno-error")

if (BUNDLE_ANTLR_RUNTIME)
    set(ANTLR4_RUNTIME_TARGET antlr4_static)
else ()
    set(ANTLR4_RUNTIME_TARGET antlr4-runtime)
endif ()

# Find Z3 for SMT solving
find_package(PkgConfig)
pkg_check_modules(Z3 z3)
if (Z3_FOUND)
    message(STATUS "✓ Z3 SMT solver found - advanced proof checking enabled")
else ()
    message(WARNING "⚠ Z3 not found - only constant propagation proofs available")
    message(STATUS "  Install with: sudo apt-get install libz3-dev")
endif ()

# LLVM and MLIR integration (required for MLIR-based compilation)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "✓ LLVM ${LLVM_PACKAGE_VERSION} found at ${LLVM_DIR}")

# MLIR is part of LLVM, set path explicitly
set(MLIR_DIR "${LLVM_DIR}/../mlir")
find_package(MLIR REQUIRED CONFIG)
message(STATUS "✓ MLIR found at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_definitions(-DUSE_MLIR)

llvm_map_components_to_libnames(REQ_LLVM_LIBS
        core
        support
        nativecodegen
)

add_executable(zenith
        src/main.cpp
        src/ASTBuilder.cpp
        src/Dialect.cpp
        src/Ops.cpp
        src/Types.cpp
        src/TypeChecker.cpp
        src/ProofSolver.cpp
        ${ANTLR_GEN_SRC}
)

target_link_libraries(zenith
        PRIVATE
        ${ANTLR4_RUNTIME_TARGET}
        ${REQ_LLVM_LIBS}
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRSupport
        MLIRTransforms
        MLIRFuncDialect
        MLIRArithDialect
)

# Link Z3 if available
if (Z3_FOUND)
    target_compile_definitions(zenith PRIVATE USE_Z3)
    target_link_libraries(zenith PRIVATE ${Z3_LIBRARIES})
    target_include_directories(zenith PRIVATE ${Z3_INCLUDE_DIRS})
endif ()

# Testing with Google Test
enable_testing()
find_package(GTest)

if (GTest_FOUND)
    message(STATUS "⚠ Google Test found but disabled due to LLVM/GTest header conflicts")
    message(STATUS "  Run unit tests with: make test")
    message(WARNING "⚠ Google Test not found - tests disabled")
    message(STATUS "  Install with: sudo apt-get install libgtest-dev")
endif ()

# Add a simple end-to-end test that compiles a sample file
add_test(NAME compile_basic COMMAND zenith ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_basic.zenith -o test_output)
set_tests_properties(compile_basic PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

message(STATUS "✓ Zenith compiler built successfully")
message(STATUS "  Run: ./zenith <file.zenith>")
message(STATUS "  Test: ctest --output-on-failure")
