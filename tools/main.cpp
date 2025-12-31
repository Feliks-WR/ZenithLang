#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "antlr4-runtime.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "ZenithParserBaseVisitor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"


// --- High-level IR (HIR) skeleton --------------------------------------------------
struct HirType {
    std::optional<int64_t> intMin;
    std::optional<int64_t> intMax;
    std::string name; // e.g., "int", "str", "bool", or array/union textual form for now
};

struct HirExpr {
    enum class Kind { IntLit, FloatLit, BoolLit, StringLit, Id, Array, Call, Group, Unsafe };
    Kind kind{};
    int64_t intVal{};
    double floatVal{};
    bool boolVal{};
    std::string strVal;
    std::vector<HirExpr> elements;          // for Array, Call args, or Group
    std::string callee;                     // for Call or Id

    static HirExpr intLit(int64_t v) { HirExpr e; e.kind = Kind::IntLit; e.intVal = v; return e; }
    static HirExpr floatLit(double v) { HirExpr e; e.kind = Kind::FloatLit; e.floatVal = v; return e; }
    static HirExpr boolLit(bool v) { HirExpr e; e.kind = Kind::BoolLit; e.boolVal = v; return e; }
    static HirExpr stringLit(std::string v) { HirExpr e; e.kind = Kind::StringLit; e.strVal = std::move(v); return e; }
    static HirExpr id(std::string v) { HirExpr e; e.kind = Kind::Id; e.callee = std::move(v); return e; }
    static HirExpr array(std::vector<HirExpr> elems) { HirExpr e; e.kind = Kind::Array; e.elements = std::move(elems); return e; }
    static HirExpr call(std::string name, std::vector<HirExpr> args) { HirExpr e; e.kind = Kind::Call; e.callee = std::move(name); e.elements = std::move(args); return e; }
    static HirExpr group(HirExpr inner) { HirExpr e; e.kind = Kind::Group; e.elements.push_back(std::move(inner)); return e; }
    static HirExpr unsafeBlock(HirExpr inner) { HirExpr e; e.kind = Kind::Unsafe; e.elements.push_back(std::move(inner)); return e; }
};

struct HirStmt {
    enum class Kind { Assign, Call };
    Kind kind{};
    std::string name;              // Assign target or call name
    std::optional<HirType> annot;  // optional type annotation for Assign
    HirExpr expr;                  // RHS or (for call) unused; args stored in callArgs
    std::vector<HirExpr> callArgs; // for Call

    static HirStmt assign(std::string lhs, std::optional<HirType> t, HirExpr rhs) {
        HirStmt s; s.kind = Kind::Assign; s.name = std::move(lhs); s.annot = std::move(t); s.expr = std::move(rhs); return s;
    }
    static HirStmt call(std::string callee, std::vector<HirExpr> args) {
        HirStmt s; s.kind = Kind::Call; s.name = std::move(callee); s.callArgs = std::move(args); return s;
    }
};

struct HirParam { std::string name; std::optional<HirType> type; };

struct HirFunc {
    enum class Kind { Proc, Subroutine, Func };
    Kind kind{};
    std::string name;
    std::vector<HirParam> params;
    std::optional<HirType> returnType;
    std::vector<HirStmt> body;          // block body
    std::optional<HirExpr> exprBody;    // "= expr" body
    bool isUnsafe = false;
};

struct HirModule {
    std::vector<HirStmt> topLevel;
    std::vector<HirFunc> functions;
};
// ------------------------------------------------------------------------------------

class HirBuilderVisitor : public ZenithParserBaseVisitor {
public:
    HirModule module;

    antlrcpp::Any visitProgram(ZenithParser::ProgramContext *ctx) override {
        for (auto* st : ctx->stmt()) buildStmt(st->simple_stmt());
        return nullptr;
    }

private:
    void buildStmt(ZenithParser::Simple_stmtContext* ctx) {
        if (!ctx) return;
        if (auto* asg = ctx->assignment()) {
            auto ann = buildType(asg->type());
            auto rhs = buildExpr(asg->expr());
            module.topLevel.push_back(HirStmt::assign(asg->ID()->getText(), ann, std::move(rhs)));
            return;
        }
        if (auto* procDecl = ctx->procedure_declaration()) { module.functions.push_back(buildFunc(procDecl)); return; }
        if (auto* subDecl = ctx->subroutine_declaration()) { module.functions.push_back(buildFunc(subDecl)); return; }
        if (auto* funDecl = ctx->function_declaration()) { module.functions.push_back(buildFunc(funDecl)); return; }
        if (auto* procCall = ctx->procedure_call()) {
            std::vector<HirExpr> args;
            for (auto* e : procCall->expr()) args.push_back(buildExpr(e));
            module.topLevel.push_back(HirStmt::call(procCall->ID()->getText(), std::move(args)));
            return;
        }
        if (auto* funCall = ctx->function_call()) {
            std::vector<HirExpr> args;
            for (auto* e : funCall->expr()) args.push_back(buildExpr(e));
            module.topLevel.push_back(HirStmt::call(funCall->ID()->getText(), std::move(args)));
            return;
        }
        if (ctx->UNSAFE()) {
            for (auto* inner : ctx->stmt()) buildStmt(inner->simple_stmt());
            return;
        }
    }

    HirFunc buildFunc(ZenithParser::Procedure_declarationContext* ctx) {
        HirFunc fn; fn.kind = HirFunc::Kind::Proc; fn.name = ctx->ID()->getText(); fn.isUnsafe = ctx->UNSAFE();
        buildParams(ctx->parameter(), fn.params);
        fn.returnType = buildType(ctx->type());
        buildBody(ctx->function_body(), fn);
        return fn;
    }
    HirFunc buildFunc(ZenithParser::Subroutine_declarationContext* ctx) {
        HirFunc fn; fn.kind = HirFunc::Kind::Subroutine; fn.name = ctx->ID()->getText(); fn.isUnsafe = ctx->UNSAFE();
        buildParams(ctx->parameter(), fn.params);
        fn.returnType = buildType(ctx->type());
        buildBody(ctx->function_body(), fn);
        return fn;
    }
    HirFunc buildFunc(ZenithParser::Function_declarationContext* ctx) {
        HirFunc fn; fn.kind = HirFunc::Kind::Func; fn.name = ctx->ID()->getText(); fn.isUnsafe = ctx->UNSAFE();
        buildParams(ctx->parameter(), fn.params);
        fn.returnType = buildType(ctx->type());
        buildBody(ctx->function_body(), fn);
        return fn;
    }

    void buildParams(const std::vector<ZenithParser::ParameterContext*>& params, std::vector<HirParam>& out) {
        for (auto* p : params) {
            HirParam hp; hp.name = p->ID()->getText(); hp.type = buildType(p->type()); out.push_back(std::move(hp));
        }
    }

    void buildBody(ZenithParser::Function_bodyContext* body, HirFunc& fn) {
        if (!body) return;
        if (body->ASSIGN()) {
            fn.exprBody = buildExpr(body->expr());
            return;
        }
        for (auto* st : body->stmt()) {
            if (auto* simp = st->simple_stmt()) {
                if (auto* asg = simp->assignment()) {
                    auto ann = buildType(asg->type());
                    auto rhs = buildExpr(asg->expr());
                    fn.body.push_back(HirStmt::assign(asg->ID()->getText(), ann, std::move(rhs)));
                } else if (auto* procCall = simp->procedure_call()) {
                    std::vector<HirExpr> args; for (auto* e : procCall->expr()) args.push_back(buildExpr(e));
                    fn.body.push_back(HirStmt::call(procCall->ID()->getText(), std::move(args)));
                } else if (auto* funCall = simp->function_call()) {
                    std::vector<HirExpr> args; for (auto* e : funCall->expr()) args.push_back(buildExpr(e));
                    fn.body.push_back(HirStmt::call(funCall->ID()->getText(), std::move(args)));
                }
            }
        }
    }

    std::optional<HirType> buildType(ZenithParser::TypeContext* ctx) {
        if (!ctx) return std::nullopt;
        HirType t; t.name = ctx->getText();
        if (ctx->range_spec()) {
            const auto ints = ctx->range_spec()->INT();
            if (ints.size() >= 2) {
                t.intMin = std::stoll(ints[0]->getText());
                t.intMax = std::stoll(ints[1]->getText());
            }
        }
        return t;
    }

    HirExpr buildExpr(ZenithParser::ExprContext* ctx) {
        if (ctx->INT()) return HirExpr::intLit(std::stoll(ctx->INT()->getText()));
        if (ctx->FLOAT()) return HirExpr::floatLit(std::stod(ctx->FLOAT()->getText()));
        if (ctx->BOOL()) return HirExpr::boolLit(ctx->BOOL()->getText() == "true");
        if (ctx->STRING()) {
            std::string s = ctx->STRING()->getText();
            if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
            return HirExpr::stringLit(std::move(s));
        }
        if (ctx->ID()) return HirExpr::id(ctx->ID()->getText());
        if (ctx->array()) {
            std::vector<HirExpr> elems; for (auto* e : ctx->array()->expr()) elems.push_back(buildExpr(e));
            return HirExpr::array(std::move(elems));
        }
        if (ctx->function_call()) {
            std::vector<HirExpr> args; for (auto* e : ctx->function_call()->expr()) args.push_back(buildExpr(e));
            return HirExpr::call(ctx->function_call()->ID()->getText(), std::move(args));
        }
        if (ctx->UNSAFE()) return HirExpr::unsafeBlock(buildExpr(ctx->expr()));
        if (ctx->expr()) return HirExpr::group(buildExpr(ctx->expr()));
        return HirExpr::id("<error>");
    }
};


// --- MLIR Lowering ---------------------------------------------------------------
static void lowerToMLIR(const HirModule& hir) {
    using namespace mlir;

    std::cerr << "=== HIR Module Info ===\n";
    std::cerr << "  Top-level statements: " << hir.topLevel.size() << "\n";
    std::cerr << "  Functions: " << hir.functions.size() << "\n";
    for (const auto& stmt : hir.topLevel) {
        if (stmt.kind == HirStmt::Kind::Assign) {
            std::cerr << "    Assign: " << stmt.name << "\n";
        } else {
            std::cerr << "    Call: " << stmt.name << "\n";
        }
    }

    // Create MLIR context, builder, and module
    MLIRContext ctx;
    ctx.loadDialect<mlir::func::FuncDialect>();
    ctx.loadDialect<mlir::arith::ArithDialect>();
    // TODO: Load ZenithDialect when it's registered properly

    std::cerr << "Creating MLIR module...\n";
    OpBuilder builder(&ctx);
    auto mlirModule = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(mlirModule.getBody());
    std::cerr << "MLIR module created.\n";

    // Lambda: lower an expression to an MLIR value
    std::function<Value(const HirExpr&, OpBuilder&, mlir::Block*)> lowerExpr;
    lowerExpr = [&](const HirExpr& e, OpBuilder& b, mlir::Block* blk) -> Value {
        b.setInsertionPointToEnd(blk);
        switch (e.kind) {
            case HirExpr::Kind::IntLit: {
                auto ty = b.getI64Type();
                return b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), ty,
                    b.getI64IntegerAttr(e.intVal));
            }
            case HirExpr::Kind::FloatLit: {
                auto ty = b.getF64Type();
                return b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), ty,
                    b.getF64FloatAttr(e.floatVal));
            }
            case HirExpr::Kind::BoolLit: {
                auto ty = b.getI1Type();
                return b.create<mlir::arith::ConstantOp>(b.getUnknownLoc(), ty,
                    b.getBoolAttr(e.boolVal));
            }
            case HirExpr::Kind::StringLit: {
                // For now, string literals are unsupported in MLIR without special ops
                // TODO: Lower to memref or a dedicated StringOp
                return nullptr;
            }
            case HirExpr::Kind::Id: {
                // Variable reference: look up in block arguments or values map
                // TODO: Implement symbol table for block scope
                return nullptr;
            }
            case HirExpr::Kind::Array: {
                // Lower array literal: create a vector or tensor
                // TODO: Implement array lowering
                return nullptr;
            }
            case HirExpr::Kind::Call: {
                // Function call: look up function and emit call op
                // TODO: Implement function call lowering
                return nullptr;
            }
            case HirExpr::Kind::Group: {
                // Group/parenthesized expression: lower inner
                if (!e.elements.empty()) return lowerExpr(e.elements[0], b, blk);
                return nullptr;
            }
            case HirExpr::Kind::Unsafe: {
                // Unsafe block: lower inner (safety attribute to be added later)
                if (!e.elements.empty()) return lowerExpr(e.elements[0], b, blk);
                return nullptr;
            }
        }
        return nullptr;
    };

    // Lambda: lower a statement
    std::function<void(const HirStmt&, OpBuilder&, mlir::Block*)> lowerStmt;
    lowerStmt = [&](const HirStmt& stmt, OpBuilder& b, mlir::Block* blk) {
        b.setInsertionPointToEnd(blk);
        switch (stmt.kind) {
            case HirStmt::Kind::Assign: {
                // Assignment: lower RHS, create a memref store or variable binding
                auto rhsVal = lowerExpr(stmt.expr, b, blk);
                if (rhsVal) {
                    // TODO: Store rhsVal under stmt.name in a symbol table
                    //       For now, we log the assignment
                }
                break;
            }
            case HirStmt::Kind::Call: {
                // Function call: emit call operation
                // TODO: Look up function, lower args, emit func::CallOp
                break;
            }
        }
    };

    // Lower top-level statements
    auto entryBlk = mlirModule.getBody();
    for (const auto& stmt : hir.topLevel) {
        lowerStmt(stmt, builder, entryBlk);
    }

    // Lower function declarations
    for (const auto& fn : hir.functions) {
        // Determine return type
        Type retTy = builder.getNoneType();
        if (fn.returnType) {
            // TODO: Parse fn.returnType.name and map to MLIR type
        }

        // Create function type
        SmallVector<Type> argTys;
        for (const auto& param : fn.params) {
            // TODO: Map param.type to MLIR type; default to i64 for now
            argTys.push_back(builder.getI64Type());
        }
        auto fnTy = builder.getFunctionType(argTys, retTy);

        // Create FuncOp
        auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), fn.name, fnTy);
        auto& body = funcOp.getBody();
        mlir::Block* bodyBlk = &body.emplaceBlock();

        // Lower function body
        if (fn.exprBody) {
            // Expression body: lower and return
            if (auto val = lowerExpr(*fn.exprBody, builder, bodyBlk)) builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), val);
        } else {
            // Block body: lower statements
            for (const auto& stmt : fn.body) {
                lowerStmt(stmt, builder, bodyBlk);
            }
            // Add implicit return for void functions
            builder.setInsertionPointToEnd(bodyBlk);
            builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
        }
    }

    // Print MLIR module to stdout
    std::cerr << "Printing MLIR module...\n";
    mlirModule.print(llvm::outs());
    std::cerr << "\n=== MLIR lowering complete ===\n";
}

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: zenith-cli <input_file>" << std::endl;
        return 1;
    }

    std::ifstream stream(argv[1]);
    antlr4::ANTLRInputStream input(stream);
    ZenithLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    ZenithParser parser(&tokens);

    // Emit parse errors to stderr for debugging
    parser.removeErrorListeners();
    parser.addErrorListener(new antlr4::ConsoleErrorListener());

    // --- Parse and lower -----------------------------------------------------------
    HirBuilderVisitor visitor;
    visitor.visit(parser.program());

    lowerToMLIR(visitor.module);

    return 0;
}
