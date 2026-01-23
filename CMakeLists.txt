cmake_minimum_required(VERSION 3.20)
project(ZenithLanguage)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    ${ANTLR_GEN_SRC}
)

target_link_libraries(zenith
    PRIVATE
    antlr4-runtime
)

message(STATUS "✓ Zenith compiler built successfully")
message(STATUS "  Run: ./zenith <file.zenith>")