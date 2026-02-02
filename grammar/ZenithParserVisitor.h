
// Generated from ../../grammar/ZenithParser.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "ZenithParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by ZenithParser.
 */
class  ZenithParserVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by ZenithParser.
   */
    virtual std::any visitProgram(ZenithParser::ProgramContext *context) = 0;

    virtual std::any visitStatement(ZenithParser::StatementContext *context) = 0;

    virtual std::any visitSemi(ZenithParser::SemiContext *context) = 0;

    virtual std::any visitVarDeclaration(ZenithParser::VarDeclarationContext *context) = 0;

    virtual std::any visitIdentifierList(ZenithParser::IdentifierListContext *context) = 0;

    virtual std::any visitFunctionDecl(ZenithParser::FunctionDeclContext *context) = 0;

    virtual std::any visitParameterList(ZenithParser::ParameterListContext *context) = 0;

    virtual std::any visitParameter(ZenithParser::ParameterContext *context) = 0;

    virtual std::any visitEquation(ZenithParser::EquationContext *context) = 0;

    virtual std::any visitExprStatement(ZenithParser::ExprStatementContext *context) = 0;

    virtual std::any visitBlockStatement(ZenithParser::BlockStatementContext *context) = 0;

    virtual std::any visitIfStatement(ZenithParser::IfStatementContext *context) = 0;

    virtual std::any visitWhileStatement(ZenithParser::WhileStatementContext *context) = 0;

    virtual std::any visitForStatement(ZenithParser::ForStatementContext *context) = 0;

    virtual std::any visitReturnStatement(ZenithParser::ReturnStatementContext *context) = 0;

    virtual std::any visitPrintStatement(ZenithParser::PrintStatementContext *context) = 0;

    virtual std::any visitType(ZenithParser::TypeContext *context) = 0;

    virtual std::any visitBaseType(ZenithParser::BaseTypeContext *context) = 0;

    virtual std::any visitDependentPredicate(ZenithParser::DependentPredicateContext *context) = 0;

    virtual std::any visitPredicate(ZenithParser::PredicateContext *context) = 0;

    virtual std::any visitRangePredicate(ZenithParser::RangePredicateContext *context) = 0;

    virtual std::any visitUnaryPredicate(ZenithParser::UnaryPredicateContext *context) = 0;

    virtual std::any visitBinaryPredicate(ZenithParser::BinaryPredicateContext *context) = 0;

    virtual std::any visitComplexPredicate(ZenithParser::ComplexPredicateContext *context) = 0;

    virtual std::any visitPredicateArgList(ZenithParser::PredicateArgListContext *context) = 0;

    virtual std::any visitPredicateValue(ZenithParser::PredicateValueContext *context) = 0;

    virtual std::any visitPredicateOp(ZenithParser::PredicateOpContext *context) = 0;

    virtual std::any visitExpression(ZenithParser::ExpressionContext *context) = 0;

    virtual std::any visitLogicalOrExpr(ZenithParser::LogicalOrExprContext *context) = 0;

    virtual std::any visitLogicalAndExpr(ZenithParser::LogicalAndExprContext *context) = 0;

    virtual std::any visitBitwiseOrExpr(ZenithParser::BitwiseOrExprContext *context) = 0;

    virtual std::any visitBitwiseXorExpr(ZenithParser::BitwiseXorExprContext *context) = 0;

    virtual std::any visitBitwiseAndExpr(ZenithParser::BitwiseAndExprContext *context) = 0;

    virtual std::any visitEqualityExpr(ZenithParser::EqualityExprContext *context) = 0;

    virtual std::any visitRelationalExpr(ZenithParser::RelationalExprContext *context) = 0;

    virtual std::any visitShiftExpr(ZenithParser::ShiftExprContext *context) = 0;

    virtual std::any visitAdditiveExpr(ZenithParser::AdditiveExprContext *context) = 0;

    virtual std::any visitMultiplicativeExpr(ZenithParser::MultiplicativeExprContext *context) = 0;

    virtual std::any visitPowerExpr(ZenithParser::PowerExprContext *context) = 0;

    virtual std::any visitUnaryExpr(ZenithParser::UnaryExprContext *context) = 0;

    virtual std::any visitCallExpr(ZenithParser::CallExprContext *context) = 0;

    virtual std::any visitCallSuffix(ZenithParser::CallSuffixContext *context) = 0;

    virtual std::any visitPrimaryExpr(ZenithParser::PrimaryExprContext *context) = 0;


};

