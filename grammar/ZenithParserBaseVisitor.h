
// Generated from ../../grammar/ZenithParser.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "ZenithParserVisitor.h"


/**
 * This class provides an empty implementation of ZenithParserVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  ZenithParserBaseVisitor : public ZenithParserVisitor {
public:

  virtual std::any visitProgram(ZenithParser::ProgramContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStatement(ZenithParser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSemi(ZenithParser::SemiContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVarDeclaration(ZenithParser::VarDeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIdentifierList(ZenithParser::IdentifierListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionDecl(ZenithParser::FunctionDeclContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameterList(ZenithParser::ParameterListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameter(ZenithParser::ParameterContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEquation(ZenithParser::EquationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExprStatement(ZenithParser::ExprStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBlockStatement(ZenithParser::BlockStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIfStatement(ZenithParser::IfStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitWhileStatement(ZenithParser::WhileStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitForStatement(ZenithParser::ForStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReturnStatement(ZenithParser::ReturnStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPrintStatement(ZenithParser::PrintStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitType(ZenithParser::TypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBaseType(ZenithParser::BaseTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDependentPredicate(ZenithParser::DependentPredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPredicate(ZenithParser::PredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRangePredicate(ZenithParser::RangePredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryPredicate(ZenithParser::UnaryPredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBinaryPredicate(ZenithParser::BinaryPredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComplexPredicate(ZenithParser::ComplexPredicateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPredicateArgList(ZenithParser::PredicateArgListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPredicateValue(ZenithParser::PredicateValueContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPredicateOp(ZenithParser::PredicateOpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpression(ZenithParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalOrExpr(ZenithParser::LogicalOrExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalAndExpr(ZenithParser::LogicalAndExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitwiseOrExpr(ZenithParser::BitwiseOrExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitwiseXorExpr(ZenithParser::BitwiseXorExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitwiseAndExpr(ZenithParser::BitwiseAndExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEqualityExpr(ZenithParser::EqualityExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRelationalExpr(ZenithParser::RelationalExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitShiftExpr(ZenithParser::ShiftExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAdditiveExpr(ZenithParser::AdditiveExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMultiplicativeExpr(ZenithParser::MultiplicativeExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPowerExpr(ZenithParser::PowerExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryExpr(ZenithParser::UnaryExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCallExpr(ZenithParser::CallExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCallSuffix(ZenithParser::CallSuffixContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPrimaryExpr(ZenithParser::PrimaryExprContext *ctx) override {
    return visitChildren(ctx);
  }


};

