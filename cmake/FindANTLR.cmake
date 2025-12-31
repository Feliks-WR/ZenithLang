# Always use local pinned ANTLR JAR to ensure generator/runtime compatibility
set(LOCAL_ANTLR_JAR "${CMAKE_SOURCE_DIR}/tools/antlr-4.13.2-complete.jar")
if(NOT EXISTS ${LOCAL_ANTLR_JAR})
    message(FATAL_ERROR "Missing ${LOCAL_ANTLR_JAR}. Run build.sh to download it.")
endif()

# Find the runtime headers and library
find_path(ANTLR_INCLUDE_DIR NAMES antlr4-runtime.h
    PATH_SUFFIXES antlr4-runtime
    HINTS
        /usr/local/include
        /usr/include
    NO_DEFAULT_PATH
)
if(NOT ANTLR_INCLUDE_DIR)
    find_path(ANTLR_INCLUDE_DIR NAMES antlr4-runtime.h PATH_SUFFIXES antlr4-runtime)
endif()

find_library(ANTLR_LIBRARY NAMES antlr4-runtime antlr4_runtime
    HINTS
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib
    NO_DEFAULT_PATH
)
if(NOT ANTLR_LIBRARY)
    find_library(ANTLR_LIBRARY NAMES antlr4-runtime antlr4_runtime)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ANTLR DEFAULT_MSG ANTLR_INCLUDE_DIR ANTLR_LIBRARY)

macro(ANTLR_TARGET Name ParserFile LexerFile)
    # Define output directory
    set(ANTLR_${Name}_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/antlr4gen")
    file(MAKE_DIRECTORY ${ANTLR_${Name}_OUTPUT_DIR})

    # Use the local pinned JAR
    set(ANTLR_COMMAND java -jar ${LOCAL_ANTLR_JAR})

    # Define expected output files
    set(ANTLR_${Name}_CXX_OUTPUTS
            ${ANTLR_${Name}_OUTPUT_DIR}/${Name}Lexer.cpp
            ${ANTLR_${Name}_OUTPUT_DIR}/${Name}Lexer.h
            ${ANTLR_${Name}_OUTPUT_DIR}/${Name}Parser.cpp
            ${ANTLR_${Name}_OUTPUT_DIR}/${Name}Parser.h
            ${ANTLR_${Name}_OUTPUT_DIR}/${Name}ParserBaseVisitor.cpp
            ${ANTLR_${Name}_OUTPUT_DIR}/${Name}ParserBaseVisitor.h
            ${ANTLR_${Name}_OUTPUT_DIR}/${Name}ParserVisitor.cpp
            ${ANTLR_${Name}_OUTPUT_DIR}/${Name}ParserVisitor.h
    )

    set(ANTLR_${Name}_INPUTS
            ${CMAKE_CURRENT_SOURCE_DIR}/${ParserFile}
            ${CMAKE_CURRENT_SOURCE_DIR}/${LexerFile}
    )

    add_custom_command(
            OUTPUT ${ANTLR_${Name}_CXX_OUTPUTS}
            COMMAND ${ANTLR_COMMAND}
            -Dlanguage=Cpp
            -visitor
            -o ${ANTLR_${Name}_OUTPUT_DIR}
            ${ANTLR_${Name}_INPUTS}
            DEPENDS ${ANTLR_${Name}_INPUTS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endmacro()