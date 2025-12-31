#ifndef ZENITH_OPS
#define ZENITH_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Define the Zenith Dialect
def Zenith_Dialect : Dialect {
    let name = "zenith";
    let summary = "A dialect for the Zenith language.";
    let description = [{
        The Zenith dialect is a high-level IR for the Zenith language,
        designed for progressive lowering to standard MLIR dialects.
    }];
    let cppNamespace = "::mlir::zenith";
}

// Base class for Zenith ops
class Zenith_Op<string mnemonic, list<Trait> traits = []> :
    Op<Zenith_Dialect, mnemonic, traits>;

// Print operation
def PrintOp : Zenith_Op<"print"> {
    let summary = "Print a variable";
    let arguments = (ins AnyType:$value);
    let assemblyFormat = "$value attr-dict `:` type($value)";
}

// Constant operation (for integers/strings)
def ConstantOp : Zenith_Op<"constant", [ConstantLike, Pure]> {
    let summary = "Zenith constant operation";
    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$res);
}

#endif // ZENITH_OPS
