#include "Zenith/ZenithDialect.h"
#include "Zenith/ZenithPasses.h"
#include "Zenith/ZenithRuntime.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Parser/Parser.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"
#include <iostream>

using namespace mlir;
using namespace mlir::zenith;

extern "C" {

struct ZenithModule {
    MLIRContext context;
    OwningOpRef<ModuleOp> module;
    OpBuilder builder;
    func::FuncOp main_func;
    std::unordered_map<std::string, Value> symbol_table;

    ZenithModule() : builder(&context)
    {
        DialectRegistry registry;
        registerAllDialects(registry);
        registry.insert<ZenithDialect>();
        registerAllToLLVMIRTranslations(registry);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        module = ModuleOp::create(builder.getUnknownLoc());

        // Create a main function to hold all operations
        builder.setInsertionPointToEnd(module->getBody());
        const auto loc = builder.getUnknownLoc();
        const auto func_type = builder.getFunctionType({}, {});
        main_func = builder.create<func::FuncOp>(loc, "main", func_type);
        main_func.setPrivate();

        // Create an entry block
        auto* entry_block = main_func.addEntryBlock();
        builder.setInsertionPointToStart(entry_block);
    }
};

void* zenith_create_module() {
    return new ZenithModule();
}

void zenith_destroy_module(void* mod) {
    delete static_cast<ZenithModule*>(mod);
}

void zenith_emit_assign(void* mod, const char* name, int64_t value) {
    auto* m = static_cast<ZenithModule*>(mod);
    const auto loc = m->builder.getUnknownLoc();

    // Create constant op for the value (builder already points inside the main function)
    auto attr = m->builder.getI64IntegerAttr(value);
    auto type = m->builder.getI64Type();
    auto const_op = m->builder.create<ConstantOp>(loc, type, attr);

    // Store in a symbol table
    m->symbol_table[name] = const_op.getResult();
}

void zenith_emit_array(void* mod, const char* name, const int64_t* elements, const size_t count) {
    auto* m = static_cast<ZenithModule*>(mod);
    const auto loc = m->builder.getUnknownLoc();

    // Create array constant - we'll use a tensor type for now
    const llvm::SmallVector<int64_t> shape = {static_cast<int64_t>(count)};
    const auto tensor_type = RankedTensorType::get(shape, m->builder.getI64Type());

    // Create dense elements attribute from the array
    const llvm::SmallVector<int64_t> values(elements, elements + count);
    const auto data_attr = DenseElementsAttr::get(tensor_type, llvm::ArrayRef(values));

    auto constOp = m->builder.create<ConstantOp>(loc, tensor_type, data_attr);

    // Store in a symbol table
    m->symbol_table[name] = constOp.getResult();
}

void zenith_emit_string(void* mod, const char* name, const char* value) {
    auto* m = static_cast<ZenithModule*>(mod);
    const auto loc = m->builder.getUnknownLoc();

    // Create string constant using StringAttr
    auto str_attr = m->builder.getStringAttr(value);
    // Use a pointer type since strings are pointers
    auto ptr_type = LLVM::LLVMPointerType::get(m->builder.getContext());

    auto const_op = m->builder.create<ConstantOp>(loc, ptr_type, str_attr);

    // Store in a symbol table
    m->symbol_table[name] = const_op.getResult();
}

void zenith_emit_print(void* module, const char* var_name) {
    auto* m = static_cast<ZenithModule*>(module);
    const auto loc = m->builder.getUnknownLoc();

    // Look up the variable in the symbol table
    if (const auto it = m->symbol_table.find(var_name); it != m->symbol_table.end()) {
        Value val = it->second;
        m->builder.create<PrintOp>(loc, val);
    } else {
        llvm::errs() << "Error: Variable '" << var_name << "' not found in symbol table\n";
    }
}

void zenith_finalize(void* mod) {
    auto* m = static_cast<ZenithModule*>(mod);

    // Add a return statement to the main function
    m->builder.create<func::ReturnOp>(m->builder.getUnknownLoc());
}

void zenith_dump(void* mod) {
    auto* m = static_cast<ZenithModule*>(mod);
    m->module->dump();
}

int zenith_execute(void* mod) {
    auto* m = static_cast<ZenithModule*>(mod);

    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Create a pass manager
    PassManager pm(m->module->getContext());

    std::cout << "\n--- MLIR Before Lowering ---\n";
    m->module->dump();

    // Lower Zenith dialect to standard dialects
    pm.addPass(createZenithToStandardPass());

    // Add optimization passes
    pm.addPass(createCanonicalizerPass());

    // Lower to LLVM dialect
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    // Run all passes
    if (failed(pm.run(*m->module))) {
        llvm::errs() << "Failed to run lowering passes\n";
        return -1;
    }

    // Create a JIT execution engine
    auto maybe_engine = ExecutionEngine::create(*m->module);
    if (!maybe_engine) {
        llvm::errs() << "Failed to create execution engine: "
                     << llvm::toString(maybe_engine.takeError()) << "\n";
        return -1;
    }

    const auto& engine = maybe_engine.get();

    // Register runtime functions
    engine->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
        llvm::orc::SymbolMap symbolMap;
        symbolMap[interner("zenith_print_i64")] = {
            llvm::orc::ExecutorAddr::fromPtr(&zenith_print_i64),
            llvm::JITSymbolFlags::Exported
        };
        symbolMap[interner("zenith_print_str")] = {
            llvm::orc::ExecutorAddr::fromPtr(&zenith_print_str),
            llvm::JITSymbolFlags::Exported
        };
        symbolMap[interner("zenith_print_array")] = {
            llvm::orc::ExecutorAddr::fromPtr(&zenith_print_array),
            llvm::JITSymbolFlags::Exported
        };
        return symbolMap;
    });

    // Look up and invoke the main function
    auto expected_f_ptr = engine->lookupPacked("main");
    if (!expected_f_ptr) {
        llvm::errs() << "Failed to lookup 'main': "
                     << llvm::toString(expected_f_ptr.takeError()) << "\n";
        return -1;
    }

    auto *main_func = reinterpret_cast<void (*)()>(*expected_f_ptr);

    std::cout << "\n--- Executing ---\n";
    main_func();
    std::cout << "\n✓ Execution completed successfully\n";

    return 0;
}

} // extern "C"
