#include "zenith/frontend/parser.hpp"
#include "antlr4-runtime.h"
#include "ZenithLexer.h"
#include "ZenithParser.h"
#include "ZenithParserBaseVisitor.h"
#include <fstream>
#include <memory>

namespace zenith::ir
{
    struct HirModule;
}

namespace zenith::frontend {

class HirBuilderVisitor final : public ZenithParserBaseVisitor {
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
        ir::HirFunc fn;
        fn.kind = ir::HirFunc::Kind::Func;
        fn.name = ctx->ID()->getText();
        fn.isUnsafe = ctx->UNSAFE();
        buildParams(ctx->parameter(), fn.params, true);  // func params are immutable
        fn.returnType = buildType(ctx->type());
        buildBody(ctx->function_body(), fn);
        return fn;
    }

    void buildParams(const std::vector<ZenithParser::ParameterContext*>& params,
                     std::vector<ir::HirParam>& out, const bool isImmutable = false) {
        for (auto* p : params) {
            ir::HirParam hp;
            hp.name = p->ID()->getText();
            hp.type = buildType(p->type());
            hp.isImmutable = isImmutable;
            out.push_back(std::move(hp));
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
        ir::HirType t;
        t.name = ctx->getText();

        // Handle dependent types: base_type '{ var_name: constraint_list }'
        if (ctx->dependent_type()) {
            auto* depType = ctx->dependent_type();
            t.isDependent = true;
            t.depVarName = depType->ID()->getText();

            // Parse constraint_list from a dependent type
            if (auto* constraintList = depType->constraint_list()) {
                parseConstraintList(constraintList, t);
            }
            return t;
        }

        // Handle inline constraints: base_type @ constraint_list
        if (ctx->dependent_constraints()) {
            auto* constraints = ctx->dependent_constraints();
            if (auto* constraintList = constraints->constraint_list()) {
                t.hasInlineConstraints = true;
                parseConstraintList(constraintList, t);
            }
            return t;
        }

        // Handle range specs with arithmetic: [expr .. expr] or (expr .. expr)
        if (ctx->range_spec()) {
            auto rangeExprs = ctx->range_spec()->expr();
            if (rangeExprs.size() >= 2) {
                auto minExpr = buildRangeExprFromExpr(rangeExprs[0]);
                auto maxExpr = buildRangeExprFromExpr(rangeExprs[1]);

                // Try to evaluate constant expressions for min/max
                if (minExpr.kind == ir::HirRangeExpr::OpKind::Lit) {
                    t.intMin = minExpr.value;
                }
                if (maxExpr.kind == ir::HirRangeExpr::OpKind::Lit) {
                    t.intMax = maxExpr.value;
                }
                t.constraints.push_back(minExpr);
                t.constraints.push_back(maxExpr);
            }
        }
        return t;
    }

    // Parse constraint_list from grammar rule
    void parseConstraintList(ZenithParser::Constraint_listContext* ctx, ir::HirType& t) {
        if (!ctx) return;
        for (const auto constraints = ctx->constraint(); auto* constraint : constraints) {
            if (constraint->ID()) {
                // Named constraint from stdlib: sorted, even, notNaN, etc.
                t.namedConstraints.push_back(constraint->ID()->getText());
            } else if (constraint->expr().size() >= 2) {
                // Range constraint: expr .. expr
                auto minExpr = buildRangeExprFromExpr(constraint->expr()[0]);
                auto maxExpr = buildRangeExprFromExpr(constraint->expr()[1]);
                t.constraints.push_back(minExpr);
                t.constraints.push_back(maxExpr);
            }
        }
    }

    // Convert Expr AST to HirRangeExpr for constraint evaluation
    ir::HirRangeExpr buildRangeExprFromExpr(ZenithParser::ExprContext* ctx) {
        if (!ctx) return ir::HirRangeExpr::lit(0);

        // Get the concat_expr which is the only child
        const auto concatExpr = ctx->concat_expr();
        if (!concatExpr) return ir::HirRangeExpr::lit(0);

        // For simplicity with literals and identifiers
        const auto addExprs = concatExpr->add_expr();
        if (addExprs.empty()) return ir::HirRangeExpr::lit(0);

        const auto addExpr = addExprs[0];
        const auto mulExprs = addExpr->mul_expr();
        if (mulExprs.empty()) return ir::HirRangeExpr::lit(0);

        const auto mulExpr = mulExprs[0];
        const auto primaries = mulExpr->primary();
        if (primaries.empty()) return ir::HirRangeExpr::lit(0);

        const auto primary = primaries[0];
        if (primary->INT()) {
            return ir::HirRangeExpr::lit(std::stoll(primary->INT()->getText()));
        }
        if (primary->ID()) {
            return ir::HirRangeExpr::id(primary->ID()->getText());
        }

        return ir::HirRangeExpr::lit(0);
    }


    std::optional<ir::HirType> buildArrayElementType(ZenithParser::Array_element_typeContext* ctx) {
        if (!ctx) return std::nullopt;

        // For now, just return a basic type
        // This can be extended in the future when array element types need more detail
        ir::HirType t;
        t.name = "element";
        return t;
    }


    ir::HirExpr buildExpr(ZenithParser::ExprContext* ctx) {
        if (!ctx) return ir::HirExpr::id("<error>");
        return buildConcatExpr(ctx->concat_expr());
    }

    ir::HirExpr buildConcatExpr(ZenithParser::Concat_exprContext* ctx) {
        if (!ctx) return ir::HirExpr::id("<error>");

        auto addExprs = ctx->add_expr();
        if (addExprs.empty()) return ir::HirExpr::id("<error>");

        ir::HirExpr result = buildAddExpr(addExprs[0]);

        // Handle CONCAT operators (the lowest precedence)
        for (size_t i = 1; i < addExprs.size(); ++i) {
            result = ir::HirExpr::binOp(ir::HirExpr::BinOpKind::Concat,
                                       result, buildAddExpr(addExprs[i]));
        }
        return result;
    }

    ir::HirExpr buildAddExpr(ZenithParser::Add_exprContext* ctx) {
        if (!ctx) return ir::HirExpr::id("<error>");

        auto mulExprs = ctx->mul_expr();
        if (mulExprs.empty()) return ir::HirExpr::id("<error>");

        ir::HirExpr result = buildMulExpr(mulExprs[0]);

        // Get all PLUS and MINUS tokens
        auto plusTokens = ctx->PLUS();
        auto minusTokens = ctx->MINUS();
        size_t plusIdx = 0, minusIdx = 0;

        // Handle addition/subtraction operators
        for (size_t i = 1; i < mulExprs.size(); ++i) {
            ir::HirExpr::BinOpKind op = ir::HirExpr::BinOpKind::Add;

            // Simplified: alternate between plus and minus, preferring plus if both exist
            if (plusIdx < plusTokens.size()) {
                op = ir::HirExpr::BinOpKind::Add;
                plusIdx++;
            } else if (minusIdx < minusTokens.size()) {
                op = ir::HirExpr::BinOpKind::Sub;
                minusIdx++;
            }

            result = ir::HirExpr::binOp(op, result, buildMulExpr(mulExprs[i]));
        }
        return result;
    }

    ir::HirExpr buildMulExpr(ZenithParser::Mul_exprContext* ctx) {
        if (!ctx) return ir::HirExpr::id("<error>");

        auto primaries = ctx->primary();
        if (primaries.empty()) return ir::HirExpr::id("<error>");

        ir::HirExpr result = buildPrimary(primaries[0]);

        // Get all STAR and SLASH tokens
        auto starTokens = ctx->STAR();
        auto slashTokens = ctx->SLASH();
        size_t starIdx = 0, slashIdx = 0;

        // Handle multiplication/division operators
        for (size_t i = 1; i < primaries.size(); ++i) {
            ir::HirExpr::BinOpKind op = ir::HirExpr::BinOpKind::Mul;

            // Simplified: alternate between star and slash, preferring star if both exist
            if (starIdx < starTokens.size()) {
                op = ir::HirExpr::BinOpKind::Mul;
                starIdx++;
            } else if (slashIdx < slashTokens.size()) {
                op = ir::HirExpr::BinOpKind::Div;
                slashIdx++;
            }

            result = ir::HirExpr::binOp(op, result, buildPrimary(primaries[i]));
        }
        return result;
    }

    ir::HirExpr buildPrimary(ZenithParser::PrimaryContext* ctx) {
        if (!ctx) return ir::HirExpr::id("<error>");

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
        if (ctx->UNSAFE() && ctx->expr()) {
            return ir::HirExpr::unsafeBlock(buildExpr(ctx->expr()));
        }
        if (ctx->LPAREN() && ctx->expr()) {
            return ir::HirExpr::group(buildExpr(ctx->expr()));
        }
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

