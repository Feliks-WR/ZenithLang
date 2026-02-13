set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Zenith)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Zenith)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Zenith)
add_public_tablegen_target(MLIRZenithPassIncGen)

add_mlir_doc(Passes ZenithPasses ./ -gen-pass-doc)

