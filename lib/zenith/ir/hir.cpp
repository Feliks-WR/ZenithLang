#include "zenith/ir/hir.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

struct HirExpr;

namespace zenith::ir {

HirExpr HirExpr::intLit(std::int64_t v) {
    HirExpr e; e.kind = Kind::IntLit; e.intVal = v; return e;
}

HirExpr HirExpr::floatLit(double v) {
    HirExpr e; e.kind = Kind::FloatLit; e.floatVal = v; return e;
}

HirExpr HirExpr::boolLit(bool v) {
    HirExpr e; e.kind = Kind::BoolLit; e.boolVal = v; return e;
}

HirExpr HirExpr::stringLit(std::string v) {
    HirExpr e; e.kind = Kind::StringLit; e.strVal = std::move(v); return e;
}

HirExpr HirExpr::id(std::string v) {
    HirExpr e; e.kind = Kind::Id; e.callee = std::move(v); return e;
}

HirExpr HirExpr::array(std::vector<HirExpr> elems) {
    HirExpr e; e.kind = Kind::Array; e.elements = std::move(elems); return e;
}

HirExpr HirExpr::call(std::string callee, std::vector<HirExpr> args) {
    HirExpr e; e.kind = Kind::Call; e.callee = std::move(callee); e.elements = std::move(args); return e;
}

HirExpr HirExpr::group(HirExpr inner) {
    HirExpr e; e.kind = Kind::Group; e.elements.push_back(std::move(inner)); return e;
}

HirExpr HirExpr::unsafeBlock(HirExpr inner) {
    HirExpr e; e.kind = Kind::Unsafe; e.elements.push_back(std::move(inner)); return e;
}

HirStmt HirStmt::assign(std::string lhs, std::optional<HirType> t, HirExpr rhs) {
    HirStmt s; s.kind = Kind::Assign; s.name = std::move(lhs); s.annot = std::move(t); s.expr = std::move(rhs); return s;
}

HirStmt HirStmt::call(std::string callee, std::vector<HirExpr> args) {
    HirStmt s; s.kind = Kind::Call; s.name = std::move(callee); s.callArgs = std::move(args); return s;
}

}  // namespace zenith::ir

