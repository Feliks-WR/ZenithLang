#include "TypeChecker.h"
#include "Types.h"
#include <gtest/gtest.h>
#include <memory>

using namespace mlir::customlang;

// Test Constraint creation and validation
TEST(ConstraintTests, RangeConstraint) {
  auto constraint = Constraint::makeRange(1, 10);
  EXPECT_EQ(constraint->kind, Constraint::Range);
  EXPECT_EQ(constraint->minValue, 1);
  EXPECT_EQ(constraint->maxValue, 10);
  EXPECT_TRUE(constraint->isValid());
  EXPECT_EQ(constraint->toString(), "{1..10}");
}

TEST(ConstraintTests, PredicateConstraint) {
  auto constraint = Constraint::makePredicate("it != 0");
  EXPECT_EQ(constraint->kind, Constraint::Predicate);
  EXPECT_TRUE(constraint->isValid());
  EXPECT_EQ(constraint->toString(), "{it != 0}");
}

TEST(ConstraintTests, NonNullConstraint) {
  auto constraint = Constraint::makeNonNull();
  EXPECT_EQ(constraint->kind, Constraint::NonNull);
  EXPECT_TRUE(constraint->isValid());
  EXPECT_EQ(constraint->toString(), "{nonnull}");
}

// Test DependentType creation
TEST(DependentTypeTests, BasicIntType) {
  auto intType = DependentType::makeInt();
  EXPECT_EQ(intType->kind, DependentType::Int);
  EXPECT_EQ(intType->toString(), "int");
  EXPECT_FALSE(intType->requiresProof());
}

TEST(DependentTypeTests, ConstrainedIntType) {
  auto nonZeroConstraint = Constraint::makePredicate("it != 0");
  auto intType = DependentType::makeIntWithConstraint(nonZeroConstraint);
  EXPECT_EQ(intType->kind, DependentType::Int);
  EXPECT_TRUE(intType->requiresProof());
  EXPECT_EQ(intType->toString(), "int{it != 0}");
}

TEST(DependentTypeTests, RangeIntType) {
  auto rangeConstraint = Constraint::makeRange(1, 31);
  auto dayType = DependentType::makeIntWithConstraint(rangeConstraint);
  EXPECT_EQ(dayType->kind, DependentType::Int);
  EXPECT_TRUE(dayType->requiresProof());
  EXPECT_EQ(dayType->toString(), "int{1..31}");
}

TEST(DependentTypeTests, PointerType) {
  auto intType = DependentType::makeInt();
  auto ptrType = DependentType::makePointer(intType);
  EXPECT_EQ(ptrType->kind, DependentType::Pointer);
  EXPECT_EQ(ptrType->toString(), "*int");
}

TEST(DependentTypeTests, NonNullPointerType) {
  auto intType = DependentType::makeInt();
  auto ptrType = DependentType::makePointer(intType);
  ptrType->constraints.push_back(Constraint::makeNonNull());
  EXPECT_TRUE(ptrType->requiresProof());
  EXPECT_EQ(ptrType->toString(), "*int{nonnull}");
}

TEST(DependentTypeTests, ArrayType) {
  auto intType = DependentType::makeInt();
  auto arrayType = DependentType::makeArray(intType, "N");
  EXPECT_EQ(arrayType->kind, DependentType::Array);
  EXPECT_EQ(arrayType->arrayLengthParam, "N");
  EXPECT_EQ(arrayType->toString(), "[int; N]");
}

TEST(DependentTypeTests, TypeCompatibility) {
  auto intType1 = DependentType::makeInt();
  auto intType2 = DependentType::makeInt();
  EXPECT_TRUE(intType1->isCompatibleWith(intType2));

  auto floatType = DependentType::makeFloat();
  EXPECT_FALSE(intType1->isCompatibleWith(floatType));
}

TEST(DependentTypeTests, ConstraintSatisfaction) {
  auto nonZeroConstraint = Constraint::makePredicate("it != 0");
  auto intType = DependentType::makeIntWithConstraint(nonZeroConstraint);

  EXPECT_TRUE(intType->satisfiesConstraints("5"));
  EXPECT_TRUE(intType->satisfiesConstraints("-1"));
  EXPECT_FALSE(intType->satisfiesConstraints("0"));
}

TEST(DependentTypeTests, RangeConstraintSatisfaction) {
  auto rangeConstraint = Constraint::makeRange(1, 10);
  auto intType = DependentType::makeIntWithConstraint(rangeConstraint);

  EXPECT_TRUE(intType->satisfiesConstraints("1"));
  EXPECT_TRUE(intType->satisfiesConstraints("5"));
  EXPECT_TRUE(intType->satisfiesConstraints("10"));
  EXPECT_FALSE(intType->satisfiesConstraints("0"));
  EXPECT_FALSE(intType->satisfiesConstraints("11"));
}

// Test TypeEnv
TEST(TypeEnvTests, AddAndGetType) {
  TypeEnv env;
  auto intType = DependentType::makeInt();

  env.addType("x", intType);
  EXPECT_TRUE(env.hasType("x"));

  auto retrieved = env.getType("x");
  EXPECT_TRUE(retrieved->isCompatibleWith(intType));
}

TEST(TypeEnvTests, ProofObligations) {
  TypeEnv env;
  auto constraint = Constraint::makePredicate("it != 0");

  env.addProofObligation("line_5", constraint);

  auto proofs = env.getProofs("line_5");
  EXPECT_EQ(proofs.size(), 1);
  EXPECT_EQ(proofs[0]->expression, "it != 0");
}

// Test TypeChecker
TEST(TypeCheckerTests, DeclareVariable) {
  TypeChecker checker;
  auto intType = DependentType::makeInt();

  checker.declareVariable("x", intType);

  auto retrieved = checker.getVariableType("x");
  EXPECT_TRUE(retrieved->isCompatibleWith(intType));
}

TEST(TypeCheckerTests, AssignValidValue) {
  TypeChecker checker;
  auto nonZeroConstraint = Constraint::makePredicate("it != 0");
  auto intType = DependentType::makeIntWithConstraint(nonZeroConstraint);

  checker.declareVariable("x", intType);
  checker.assignVariable("x", "5");

  EXPECT_FALSE(checker.hasErrors());
}

TEST(TypeCheckerTests, AssignInvalidValue) {
  TypeChecker checker;
  auto nonZeroConstraint = Constraint::makePredicate("it != 0");
  auto intType = DependentType::makeIntWithConstraint(nonZeroConstraint);

  checker.declareVariable("x", intType);
  checker.assignVariable("x", "0");

  EXPECT_TRUE(checker.hasErrors());
}

TEST(TypeCheckerTests, ArrayAccess) {
  TypeChecker checker;
  auto intType = DependentType::makeInt();
  auto arrayType = DependentType::makeArray(intType, "10");

  checker.declareVariable("arr", arrayType);
  checker.checkArrayAccess(arrayType, "0", "line_10");

  auto unsatisfied = checker.getUnsatisfiedObligations();
  EXPECT_EQ(unsatisfied.size(), 1);
  EXPECT_EQ(unsatisfied[0].kind, ProofObligation::ArrayBounds);
}

TEST(TypeCheckerTests, DivisionByZero) {
  TypeChecker checker;
  auto intType = DependentType::makeInt();

  checker.declareVariable("x", intType);
  checker.checkDivision(intType, "y", "line_15");

  auto unsatisfied = checker.getUnsatisfiedObligations();
  EXPECT_EQ(unsatisfied.size(), 1);
  EXPECT_EQ(unsatisfied[0].kind, ProofObligation::DivisionNonZero);
}

TEST(TypeCheckerTests, DivisionWithNonZeroConstraint) {
  TypeChecker checker;
  auto nonZeroConstraint = Constraint::makePredicate("(!=0)");
  auto intType = DependentType::makeIntWithConstraint(nonZeroConstraint);

  checker.declareVariable("x", intType);
  checker.checkDivision(intType, "x", "line_20");

  auto unsatisfied = checker.getUnsatisfiedObligations();
  EXPECT_EQ(unsatisfied.size(), 0);
}

TEST(TypeCheckerTests, PointerDereference) {
  TypeChecker checker;
  auto intType = DependentType::makeInt();
  auto ptrType = DependentType::makePointer(intType);

  checker.checkPointerDereference(ptrType, "line_25");

  auto unsatisfied = checker.getUnsatisfiedObligations();
  EXPECT_EQ(unsatisfied.size(), 1);
  EXPECT_EQ(unsatisfied[0].kind, ProofObligation::PointerDeref);
}

TEST(TypeCheckerTests, NonNullPointerDereference) {
  TypeChecker checker;
  auto intType = DependentType::makeInt();
  auto ptrType = DependentType::makePointer(intType);
  ptrType->constraints.push_back(Constraint::makeNonNull());

  checker.checkPointerDereference(ptrType, "line_30");

  auto unsatisfied = checker.getUnsatisfiedObligations();
  EXPECT_EQ(unsatisfied.size(), 0);
}

TEST(TypeCheckerTests, ClearEnvironment) {
  TypeChecker checker;
  auto intType = DependentType::makeInt();

  checker.declareVariable("x", intType);
  EXPECT_TRUE(checker.getTypeEnv().hasType("x"));

  checker.clear();
  EXPECT_FALSE(checker.getTypeEnv().hasType("x"));
}
