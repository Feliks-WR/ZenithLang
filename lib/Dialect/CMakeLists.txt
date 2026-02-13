add_mlir_dialect_library(MLIRZenithDialect
    ZenithDialect.cpp
    ZenithOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Zenith

    DEPENDS
    MLIRZenithOpsIncGen
    MLIRZenithDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRFunctionInterfaces
    MLIRSideEffectInterfaces
    MLIRCallInterfaces
    MLIRCastInterfaces
)

