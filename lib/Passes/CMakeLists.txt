add_mlir_library(MLIRZenithPasses
    Passes.cpp
    LowerToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Zenith/Passes

    DEPENDS
    MLIRZenithPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRZenithDialect
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRFuncDialect
)

