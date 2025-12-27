cmake_minimum_required(VERSION 4.1.2)
project(Zenith LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

# --- MLIR/LLVM Setup ---
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Load LLVM and MLIR specific CMake modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
)
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# --- TableGen for Zenith Dialect ---
set(LLVM_TARGET_DEFINITIONS include/Zenith/ZenithOps.td)
mlir_tablegen(include/Zenith/ZenithOpsDialect.h.inc -gen-dialect-decls -dialect=zenith)
mlir_tablegen(include/Zenith/ZenithOpsDialect.cpp.inc -gen-dialect-defs -dialect=zenith)
mlir_tablegen(include/Zenith/ZenithOps.h.inc -gen-op-decls)
mlir_tablegen(include/Zenith/ZenithOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(ZenithOpsIncGen)

# --- Zenith Dialect Library ---
add_library(ZenithDialect SHARED
        lib/Zenith/ZenithDialect.cpp
        lib/Zenith/ZenithBridge.cpp
        lib/Zenith/ZenithToStandard.cpp
        lib/Zenith/ZenithRuntime.cpp
)
add_dependencies(ZenithDialect ZenithOpsIncGen)
target_link_libraries(ZenithDialect
        PUBLIC
        MLIR
        MLIRSideEffectInterfaces
        MLIRArithDialect
        MLIRFuncDialect
        MLIRTransforms
)
