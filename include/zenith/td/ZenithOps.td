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
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

// Base class for Zenith ops
class Zenith_Op<string mnemonic, list<Trait> traits = []> :
    Op<Zenith_Dialect, mnemonic, traits>;

// Print operation
def PrintOp : Zenith_Op<"print"> {
    let summary = "Print a variable";
    let description = "Prints the SSA value to the host output.";
    let arguments = (ins AnyType:$value);
    let assemblyFormat = "$value attr-dict `:` type($value)";
}

// Constant operation (for integers/strings)
def ConstantOp : Zenith_Op<"constant", [ConstantLike, Pure]> {
    let summary = "Zenith constant operation";
    let description = "Materializes a constant attribute as an SSA value.";
    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$res);
    let assemblyFormat = "$value attr-dict `:` type($res)";
}

def AddOp : Zenith_Op<"add", [Commutative, Associative, Pure]> {
    let summary = "Addition operation";
    let description = "Adds two numeric SSA values.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs, $rhs attr-dict `:` type($result)";
}

def SubOp : Zenith_Op<"sub", [Pure]> {
    let summary = "Subtraction operation";
    let description = "Subtracts the second numeric SSA value from the first.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs, $rhs attr-dict `:` type($result)";
}

def MulOp : Zenith_Op<"mul", [Commutative, Associative, Pure]> {
    let summary = "Multiplication operation";
    let description = "Multiplies two numeric SSA values.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs, $rhs attr-dict `:` type($result)";
}

def DivOp : Zenith_Op<"div", [Pure]> {
    let summary = "Division operation";
    let description = "Divides the first numeric SSA value by the second.";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs, $rhs attr-dict `:` type($result)";
}

#endif // ZENITH_OPS
