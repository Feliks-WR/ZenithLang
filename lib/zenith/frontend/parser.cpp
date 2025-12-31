#include "zenith/frontend/parser.hpp"
#include "antlr4-runtime.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "ZenithParserBaseVisitor.h"
#include <fstream>
#include <memory>
#include <sstream>

namespace zenith::ir
{
    struct HirModule;
}

namespace zenith::frontend {

class HirBuilderVisitor : public ZenithParserBaseVisitor {
public:
    std::unique_ptr<ir::HirModule> module = std::make_unique<ir::HirModule>();

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
            module->topLevel.push_back(ir::HirStmt::assign(asg->ID()->getText(), ann, std::move(rhs)));
            return;
        }
        if (auto* procDecl = ctx->procedure_declaration()) {
            module->functions.push_back(buildFunc(procDecl)); return;
        }
        if (auto* subDecl = ctx->subroutine_declaration()) {
            module->functions.push_back(buildFunc(subDecl)); return;
        }
        if (auto* funDecl = ctx->function_declaration()) {
            module->functions.push_back(buildFunc(funDecl)); return;
        }
        if (auto* procCall = ctx->procedure_call()) {
            std::vector<ir::HirExpr> args;
            for (auto* e : procCall->expr()) args.push_back(buildExpr(e));
            module->topLevel.push_back(ir::HirStmt::call(procCall->ID()->getText(), std::move(args)));
            return;
        }
        if (auto* funCall = ctx->function_call()) {
            std::vector<ir::HirExpr> args;
            for (auto* e : funCall->expr()) args.push_back(buildExpr(e));
            module->topLevel.push_back(ir::HirStmt::call(funCall->ID()->getText(), std::move(args)));
            return;
        }
    }

    ir::HirFunc buildFunc(ZenithParser::Procedure_declarationContext* ctx) {
        ir::HirFunc fn; fn.kind = ir::HirFunc::Kind::Proc; fn.name = ctx->ID()->getText(); fn.isUnsafe = ctx->UNSAFE();
        buildParams(ctx->parameter(), fn.params);
        fn.returnType = buildType(ctx->type());
        buildBody(ctx->function_body(), fn);
        return fn;
    }
    ir::HirFunc buildFunc(ZenithParser::Subroutine_declarationContext* ctx) {
        ir::HirFunc fn; fn.kind = ir::HirFunc::Kind::Subroutine; fn.name = ctx->ID()->getText(); fn.isUnsafe = ctx->UNSAFE();
        buildParams(ctx->parameter(), fn.params);
        fn.returnType = buildType(ctx->type());
        buildBody(ctx->function_body(), fn);
        return fn;
    }
    ir::HirFunc buildFunc(ZenithParser::Function_declarationContext* ctx) {
        ir::HirFunc fn; fn.kind = ir::HirFunc::Kind::Func; fn.name = ctx->ID()->getText(); fn.isUnsafe = ctx->UNSAFE();
        buildParams(ctx->parameter(), fn.params);
        fn.returnType = buildType(ctx->type());
        buildBody(ctx->function_body(), fn);
        return fn;
    }

    void buildParams(const std::vector<ZenithParser::ParameterContext*>& params,
                     std::vector<ir::HirParam>& out) {
        for (auto* p : params) {
            ir::HirParam hp; hp.name = p->ID()->getText(); hp.type = buildType(p->type()); out.push_back(std::move(hp));
        }
    }

    void buildBody(ZenithParser::Function_bodyContext* body, ir::HirFunc& fn) {
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
                    fn.body.push_back(ir::HirStmt::assign(asg->ID()->getText(), ann, std::move(rhs)));
                } else if (auto* procCall = simp->procedure_call()) {
                    std::vector<ir::HirExpr> args; for (auto* e : procCall->expr()) args.push_back(buildExpr(e));
                    fn.body.push_back(ir::HirStmt::call(procCall->ID()->getText(), std::move(args)));
                } else if (auto* funCall = simp->function_call()) {
                    std::vector<ir::HirExpr> args; for (auto* e : funCall->expr()) args.push_back(buildExpr(e));
                    fn.body.push_back(ir::HirStmt::call(funCall->ID()->getText(), std::move(args)));
                }
            }
        }
    }

    std::optional<ir::HirType> buildType(ZenithParser::TypeContext* ctx) {
        if (!ctx) return std::nullopt;
        ir::HirType t; t.name = ctx->getText();
        if (ctx->range_spec()) {
            const auto ints = ctx->range_spec()->INT();
            if (ints.size() >= 2) {
                t.intMin = std::stoll(ints[0]->getText());
                t.intMax = std::stoll(ints[1]->getText());
            }
        }
        return t;
    }

    ir::HirExpr buildExpr(ZenithParser::ExprContext* ctx) {
        if (ctx->INT()) return ir::HirExpr::intLit(std::stoll(ctx->INT()->getText()));
        if (ctx->FLOAT()) return ir::HirExpr::floatLit(std::stod(ctx->FLOAT()->getText()));
        if (ctx->BOOL()) return ir::HirExpr::boolLit(ctx->BOOL()->getText() == "true");
        if (ctx->STRING()) {
            std::string s = ctx->STRING()->getText();
            if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
                s = s.substr(1, s.size() - 2);
            return ir::HirExpr::stringLit(std::move(s));
        }
        if (ctx->ID()) return ir::HirExpr::id(ctx->ID()->getText());
        if (ctx->array()) {
            std::vector<ir::HirExpr> elems;
            for (auto* e : ctx->array()->expr()) elems.push_back(buildExpr(e));
            return ir::HirExpr::array(std::move(elems));
        }
        if (ctx->function_call()) {
            std::vector<ir::HirExpr> args;
            for (auto* e : ctx->function_call()->expr()) args.push_back(buildExpr(e));
            return ir::HirExpr::call(ctx->function_call()->ID()->getText(), std::move(args));
        }
        if (ctx->UNSAFE()) return ir::HirExpr::unsafeBlock(buildExpr(ctx->expr()));
        if (ctx->expr()) return ir::HirExpr::group(buildExpr(ctx->expr()));
        return ir::HirExpr::id("<error>");
    }
};

std::unique_ptr<ir::HirModule> Parser::parseFile(const std::string& filename) {
    std::ifstream stream(filename);
    if (!stream) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    antlr4::ANTLRInputStream input(stream);
    ZenithLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    ZenithParser parser(&tokens);

    parser.removeErrorListeners();
    parser.addErrorListener(new antlr4::ConsoleErrorListener());

    HirBuilderVisitor visitor;
    visitor.visit(parser.program());

    return std::move(visitor.module);
}

std::unique_ptr<ir::HirModule> Parser::parseString(const std::string& source) {
    antlr4::ANTLRInputStream input(source);
    ZenithLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    ZenithParser parser(&tokens);

    parser.removeErrorListeners();
    parser.addErrorListener(new antlr4::ConsoleErrorListener());

    HirBuilderVisitor visitor;
    visitor.visit(parser.program());

    return std::move(visitor.module);
}

}  // namespace zenith::frontend

