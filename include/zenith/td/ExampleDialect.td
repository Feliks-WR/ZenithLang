#ifndef ZENITH_EXAMPLE_DIALECT
#define ZENITH_EXAMPLE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Copy/rename this file to start a new dialect definition.
def Example_Dialect : Dialect {
  let name = "example";
  let summary = "Example dialect skeleton";
  let description = [{
    Skeleton dialect to copy when defining a new dialect. Rename the
    dialect, namespace, and ops to match your domain.
  }];
  let cppNamespace = "::mlir::example";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Convenience base for ops in this dialect.
class Example_Op<string mnemonic, list<Trait> traits = []> :
    Op<Example_Dialect, mnemonic, traits>;

// Sample op: constant.
def Example_ConstantOp : Example_Op<"constant", [ConstantLike, Pure]> {
  let summary = "Materialize a constant";
  let description = "Creates an SSA value from an attribute.";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$value attr-dict `:` type($res)";
}

// Sample op: print.
def Example_PrintOp : Example_Op<"print"> {
  let summary = "Print a value";
  let description = "Emits the SSA value to an external sink.";
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // ZENITH_EXAMPLE_DIALECT

