set(LLVM_TARGET_DEFINITIONS ZenithDialect.td)
mlir_tablegen(ZenithOpsDialect.h.inc -gen-dialect-decls -dialect=zenith)
mlir_tablegen(ZenithOpsDialect.cpp.inc -gen-dialect-defs -dialect=zenith)
add_public_tablegen_target(MLIRZenithDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ZenithOps.td)
mlir_tablegen(ZenithOps.h.inc -gen-op-decls)
mlir_tablegen(ZenithOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRZenithOpsIncGen)

add_mlir_doc(ZenithDialect ZenithDialect Zenith/ -gen-dialect-doc)
add_mlir_doc(ZenithOps ZenithOps Zenith/ -gen-op-doc)

