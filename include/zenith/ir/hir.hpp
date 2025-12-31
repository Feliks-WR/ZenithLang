#ifndef ZENITH_IR_HIR_HPP
#define ZENITH_IR_HIR_HPP

#include <optional>
#include <string>
#include <vector>
#include <memory>

namespace zenith::ir {

struct HirType {
    std::optional<int64_t> intMin;
    std::optional<int64_t> intMax;
    std::string name;
};

struct HirExpr {
    enum class Kind { IntLit, FloatLit, BoolLit, StringLit, Id, Array, Call, Group, Unsafe };

    Kind kind{};
    int64_t intVal{};
    double floatVal{};
    bool boolVal{};
    std::string strVal;
    std::vector<HirExpr> elements;
    std::string callee;

    static HirExpr intLit(int64_t v);
    static HirExpr floatLit(double v);
    static HirExpr boolLit(bool v);
    static HirExpr stringLit(std::string v);
    static HirExpr id(std::string v);
    static HirExpr array(std::vector<HirExpr> elems);
    static HirExpr call(std::string name, std::vector<HirExpr> args);
    static HirExpr group(HirExpr inner);
    static HirExpr unsafeBlock(HirExpr inner);
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

struct HirParam {
    std::string name;
    std::optional<HirType> type;
};

struct HirFunc {
    enum class Kind { Proc, Subroutine, Func };

    Kind kind{};
    std::string name;
    std::vector<HirParam> params;
    std::optional<HirType> returnType;
    std::vector<HirStmt> body;
    std::optional<HirExpr> exprBody;
    bool isUnsafe = false;
};

struct HirModule {
    std::vector<HirStmt> topLevel;
    std::vector<HirFunc> functions;
};

}  // namespace zenith::ir

#endif  // ZENITH_IR_HIR_HPP

