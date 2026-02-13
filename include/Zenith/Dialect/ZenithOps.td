//===- ZenithOps.td - Zenith dialect ops -------------------*- tablegen -*-===//
//
// This file defines the basic operations for the Zenith dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ZENITH_OPS
#define ZENITH_OPS

include "Zenith/Dialect/ZenithDialect.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Zenith Types
//===----------------------------------------------------------------------===//

def Zenith_Type : Type<CPred<"::llvm::isa<::mlir::zenith::ZenithType>($_self)">,
                        "Zenith type">;

//===----------------------------------------------------------------------===//
// Constant operation
//===----------------------------------------------------------------------===//

def ConstantOp : Zenith_Op<"constant", [Pure, ConstantLike]> {
    let summary = "constant value";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is
        attached to the operation as an attribute.

        Example:
        ```mlir
        %0 = zenith.constant 42 : i32
        %1 = zenith.constant 3.14 : f64
        ```
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$value attr-dict `:` type($result)";

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

def AddOp : Zenith_Op<"add", [Pure, SameOperandsAndResultType]> {
    let summary = "integer or float addition";
    let description = [{
        The `add` operation performs element-wise addition between two values.

        Example:
        ```mlir
        %2 = zenith.add %0, %1 : i32
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";

    let hasFolder = 1;
}

def SubOp : Zenith_Op<"sub", [Pure, SameOperandsAndResultType]> {
    let summary = "integer or float subtraction";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
    let hasFolder = 1;
}

def MulOp : Zenith_Op<"mul", [Pure, SameOperandsAndResultType]> {
    let summary = "integer or float multiplication";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
    let hasFolder = 1;
}

def DivOp : Zenith_Op<"div", [Pure, SameOperandsAndResultType]> {
    let summary = "integer or float division";
    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Function operations
//===----------------------------------------------------------------------===//

def FuncOp : Zenith_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove, Symbol
]> {
    let summary = "user defined function";
    let description = [{
        The `zenith.func` operation represents a user defined function. These
        can be referenced by name in the module and can be called with the
        `zenith.call` operation.

        Example:
        ```mlir
        zenith.func @add(%arg0: i32, %arg1: i32) -> i32 {
            %0 = zenith.add %arg0, %arg1 : i32
            zenith.return %0 : i32
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
        "StringRef":$name, "FunctionType":$type,
        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        Region *getCallableRegion() { return &getBody(); }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def ReturnOp : Zenith_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let summary = "return operation";
    let description = [{
        The `return` operation represents a return from a function. The operation
        takes variable number of operands and produces no results.

        Example:
        ```mlir
        zenith.return %0 : i32
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$operands);
    let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def CallOp : Zenith_Op<"call", [CallOpInterface]> {
    let summary = "call operation";
    let description = [{
        The `call` operation represents a direct call to a function. The operands
        and result types must match the specified function signature.

        Example:
        ```mlir
        %2 = zenith.call @add(%0, %1) : (i32, i32) -> i32
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let extraClassDeclaration = [{
        /// Get the argument operands to the called function.
        operand_range getArgOperands() { return getOperands(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }
    }];
}

//===----------------------------------------------------------------------===//
// Print operation
//===----------------------------------------------------------------------===//

def PrintOp : Zenith_Op<"print"> {
    let summary = "print operation";
    let description = [{
        The `print` operation prints a value to stdout.

        Example:
        ```mlir
        zenith.print %0 : i32
        ```
    }];

    let arguments = (ins AnyType:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // ZENITH_OPS

