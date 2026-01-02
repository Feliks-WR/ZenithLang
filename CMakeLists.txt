cmake_minimum_required(VERSION 3.20)
project(Zenith LANGUAGES CXX C)

# Enable compile_commands.json for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler flags for better header discovery
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/include/antlr4-runtime")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/include/antlr4-runtime")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_BINARY_DIR}/antlr4gen")

# --- MLIR/LLVM Setup ---
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Add the local cmake directory to the module path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Allow overriding ANTLR runtime location via environment variables (Option A) BEFORE find_package(ANTLR)
if(DEFINED ENV{ANTLR_ROOT})
  set(ANTLR_ROOT $ENV{ANTLR_ROOT})
endif()
if(DEFINED ENV{ANTLR_INCLUDE_DIR})
  set(ANTLR_INCLUDE_DIR $ENV{ANTLR_INCLUDE_DIR} CACHE PATH "ANTLR include dir" FORCE)
endif()
if(DEFINED ENV{ANTLR_LIBRARY})
  set(ANTLR_LIBRARY $ENV{ANTLR_LIBRARY} CACHE FILEPATH "ANTLR runtime library" FORCE)
endif()

include(FindANTLR)

# Load LLVM and MLIR specific CMake modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(
        # ANTLR4 runtime headers (local third_party first, then system)
        ${PROJECT_SOURCE_DIR}/third_party/antlr4/runtime/Cpp/runtime/src
        /usr/include/antlr4-runtime
        /usr/local/include/antlr4-runtime
        /usr/local/include
        ${ANTLR_INCLUDE_DIR}

        # LLVM/MLIR headers
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        /usr/lib/llvm-20/include
        /usr/include/llvm-20

        # Project headers
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/zenith

        # Generated headers
        ${CMAKE_CURRENT_BINARY_DIR}/antlr4gen
        ${CMAKE_CURRENT_BINARY_DIR}/include

        # System includes
        /usr/include/c++/14
        /usr/include/x86_64-linux-gnu/c++/14
        /usr/include/c++/14/backward
        /usr/lib/gcc/x86_64-linux-gnu/14/include
        /usr/include/x86_64-linux-gnu
        /usr/include
)
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Generate Lexer/Parser from split .g4 files EARLY so headers are available
antlr_target(Zenith grammar/ZenithParser.g4 grammar/ZenithLexer.g4)

# --- TableGen for Zenith Dialect ---
set(LLVM_TARGET_DEFINITIONS include/zenith/ZenithOps.td)
mlir_tablegen(include/Zenith/ZenithOpsDialect.h.inc -gen-dialect-decls -dialect=zenith)
mlir_tablegen(include/Zenith/ZenithOpsDialect.cpp.inc -gen-dialect-defs -dialect=zenith)
mlir_tablegen(include/Zenith/ZenithOps.h.inc -gen-op-decls)
mlir_tablegen(include/Zenith/ZenithOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(ZenithOpsIncGen)

# --- Zenith Dialect Library (disabled temporarily - has header issues)
# add_library(ZenithDialect SHARED
#         lib/zenith/ZenithDialect.cpp
#         lib/zenith/ZenithBridge.cpp
#         lib/zenith/ZenithToStandard.cpp
#         lib/zenith/ZenithRuntime.cpp
# )
#
# # Add dependencies
# add_dependencies(ZenithDialect ZenithOpsIncGen)
#
# # Add ANTLR include directories for dialect lib
# target_include_directories(ZenithDialect PRIVATE
#     ${ANTLR_INCLUDE_DIR}
#     ${CMAKE_CURRENT_BINARY_DIR}/antlr4gen
# )
#
# target_link_libraries(ZenithDialect
#         PUBLIC
#         MLIR
#         MLIRSideEffectInterfaces
#         MLIRArithDialect
#         MLIRFuncDialect
#         MLIRTransforms
# )
# Refactored C++ Driver with modular architecture
add_executable(zenith-cli
    ${ANTLR_Zenith_CXX_OUTPUTS}
    tools/zenith-cli.cpp
    lib/zenith/ir/hir.cpp
    lib/zenith/ir/constraint_validator.cpp
    lib/zenith/ir/immutability_checker.cpp
    lib/zenith/frontend/parser.cpp
    lib/zenith/backend/mlir_pass.cpp
    lib/zenith/backend/execution_pass.cpp
    lib/zenith/stdlib_registry.cpp
)

# Include directories for zenith-cli (comprehensive)
target_include_directories(zenith-cli PRIVATE
    # ANTLR4 includes (local third_party first for 4.13.2 compatibility)
    ${PROJECT_SOURCE_DIR}/third_party/antlr4/runtime/Cpp/runtime/src
    /usr/local/include/antlr4-runtime
    /usr/local/include
    /usr/include/antlr4-runtime
    ${ANTLR_INCLUDE_DIR}
    ${ANTLR_INCLUDE_DIR}/..

    # Generated ANTLR files
    ${CMAKE_CURRENT_BINARY_DIR}/antlr4gen

    # MLIR/LLVM includes
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    /usr/lib/llvm-20/include

    # Project includes
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/zenith
    ${PROJECT_BINARY_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include

    # System includes (explicit)
    /usr/include/c++/14
    /usr/include/x86_64-linux-gnu/c++/14
    /usr/include/c++/14/backward
    /usr/lib/gcc/x86_64-linux-gnu/14/include
    /usr/include/x86_64-linux-gnu
    /usr/include
)

target_link_libraries(zenith-cli
    ${ANTLR_LIBRARY}
    MLIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRExecutionEngine
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRTargetLLVMIRExport
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    LLVMSupport
    LLVMCore
)

