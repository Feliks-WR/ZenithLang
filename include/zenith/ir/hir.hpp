#ifndef ZENITH_IR_HIR_HPP
#define ZENITH_IR_HIR_HPP

#include <optional>
#include <string>
#include <vector>
#include <memory>

namespace zenith::ir {

// Forward declares for proper ordering
struct HirRangeExpr;
struct HirType;
struct HirExpr;
struct HirParam;

struct HirRangeExpr {
    enum class OpKind { Lit, Id, Add, Sub, Mul, Div };

    OpKind kind = OpKind::Lit;
    int64_t value = 0;
    std::string varName;
    std::shared_ptr<HirRangeExpr> left;
    std::shared_ptr<HirRangeExpr> right;

    static HirRangeExpr lit(int64_t v);
    static HirRangeExpr id(std::string name);
    static HirRangeExpr binOp(OpKind op, HirRangeExpr lhs, HirRangeExpr rhs);
};

struct HirType {
    std::optional<int64_t> intMin;
    std::optional<int64_t> intMax;
    std::string name;

    // Dependent type support
    std::string depVarName;  // Variable name for dependent type, e.g., "n" in int{n: 1..100}
    std::vector<HirRangeExpr> constraints;  // Range constraints
    std::vector<std::string> namedConstraints;  // Named constraints from stdlib: "sorted", "even", "notNaN", etc.
    bool isDependent = false;
    bool hasInlineConstraints = false;  // true if constraints are inline (int @ sorted)
};

struct HirExpr {
    enum class Kind { IntLit, FloatLit, BoolLit, StringLit, Id, Array, Call, Group, Unsafe, BinOp };
    enum class BinOpKind { Add, Sub, Mul, Div, Concat, Power };

    Kind kind{};
    int64_t intVal{};
    double floatVal{};
    bool boolVal{};
    std::string strVal;
    std::vector<HirExpr> elements;
    std::string callee;
    BinOpKind binOpKind = BinOpKind::Add;
    std::shared_ptr<HirExpr> lhs;
    std::shared_ptr<HirExpr> rhs;

    static HirExpr intLit(int64_t v);
    static HirExpr floatLit(double v);
    static HirExpr boolLit(bool v);
    static HirExpr stringLit(std::string v);
    static HirExpr id(std::string v);
    static HirExpr array(std::vector<HirExpr> elems);
    static HirExpr call(std::string name, std::vector<HirExpr> args);
    static HirExpr group(HirExpr inner);
    static HirExpr unsafeBlock(HirExpr inner);
    static HirExpr binOp(BinOpKind op, HirExpr lhs, HirExpr rhs);
};

struct HirParam {
    std::string name;
    std::optional<HirType> type;
    bool isImmutable = false;  // true for func/pure/math, false for proc
};

struct HirStmt {
    enum class Kind { Assign, Call };

    Kind kind{};
    std::string name;
    std::optional<HirType> annot;
    HirExpr expr;
    std::vector<HirExpr> callArgs;

    static HirStmt assign(std::string lhs, std::optional<HirType> t, HirExpr rhs);
    static HirStmt call(std::string callee, std::vector<HirExpr> args);
};

struct HirFunc {
    enum class Kind { Proc, Subroutine, Func, Fn };
    enum class Purity { None, Func, Pure, Math };  // None for proc/subroutine

    Kind kind{};
    Purity purity = Purity::None;
    std::string name;
    std::vector<HirParam> params;
    std::optional<HirType> returnType;
    std::optional<HirType> paramType;  // For type signature declarations
    std::vector<HirStmt> body;
    std::optional<HirExpr> exprBody;
    std::vector<std::string> effects;  // Effect/monad annotations (IO, Exception, etc.)
    bool isUnsafe = false;
    bool isTypeSignatureOnly = false;  // true for fn name : type -> type
};

struct HirModule {
    std::vector<HirStmt> topLevel;
    std::vector<HirFunc> functions;
};

}  // namespace zenith::ir

#endif  // ZENITH_IR_HIR_HPP

