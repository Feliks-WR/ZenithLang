get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(zenith-opt zenith-opt.cpp)

llvm_update_compile_flags(zenith-opt)

target_link_libraries(zenith-opt
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRZenithDialect
    MLIRZenithPasses
    MLIROptLib
    MLIRSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRArithDialect
    MLIRFuncDialect
)

mlir_check_all_link_libraries(zenith-opt)

