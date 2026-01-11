# Zenith Language Specification (draft)

> Source of truth files: `grammar/ZenithLexer.g4`, `grammar/ZenithParser.g4`, dialect ODS in `include/zenith/td/ZenithOps.td`.

## Goals and Scope
- Define syntax, static semantics, and dynamic semantics for the core Zenith language.
- Keep the spec terse and reviewable; EBNF for syntax, brief rules for semantics.
- Align with the current IR surface (Zenith dialect ops: constant/print/add/sub/mul/div).

## Lexical Grammar (summary)
- Identifiers: UTF-8 letters/digits/underscore, starting with a letter or underscore (align lexer to allow Unicode categories `Lu|Ll|Lt|Lm|Lo|Nl` for leading, `Lu|Ll|Lt|Lm|Lo|Nl|Nd|Pc` for subsequent). Example regex (PCRE-ish): `[\p{L}_][\p{L}\p{N}_]*`
- Integer literals: `[0-9]+`
- String literals: '"' (any non-quote/non-newline)* '"'
- Whitespace/comments: skipped (as in the lexer).

## Concrete Syntax (EBNF)
EBNF here is human-facing; the authoritative parser is the ANTLR grammar. Grouping `()`, choice `|`, optional `[]`, repetition `{}`.

```ebnf
Program    = { Stmt } ;
Stmt       = LetStmt | ExprStmt | PrintStmt ;
LetStmt    = "let" Identifier "=" Expr ";" ;
ExprStmt   = Expr ";" ;
PrintStmt  = "print" Expr ";" ;

Expr       = AddExpr ;
AddExpr    = MulExpr { ("+" | "-") MulExpr } ;
MulExpr    = Primary { ("*" | "/") Primary } ;
Primary    = Number | String | Identifier | "(" Expr ")" ;

Number     = Digit { Digit } ;
Digit      = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
String     = '"' { Character } '"' ;
Identifier = Letter { Letter | Digit | "_" } ;
Letter     = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" |
             "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z" |
             "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" |
             "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z" | "_" ;
```

Notes:
- Identifiers are UTF-8; update the lexer to use Unicode classes (e.g., `[:alpha:]`/`[:alnum:]` with a Unicode flag) instead of ASCII ranges above. The EBNF uses ASCII for readability only.
- Precedence/associativity come from the structure (`AddExpr` over `MulExpr`, left-associative via repetition).
- Extend with control flow or functions by adding productions; keep precedence obvious by structure.

## Static Semantics (sketch)
- Variables must be declared before use (`LetStmt` binds, `Identifier` in `Primary` refers).
- Arithmetic ops require both operands to be numeric; result type matches operands. Mixed-type rules TBD (currently homogeneous).
- Division by zero is undefined; optional static check may reject literal zero divisors.
- Strings are not valid operands for arithmetic.
- `print` accepts any type; for non-primitive types define printing rules or reject.

## Dynamic Semantics (sketch)
- Evaluation is eager, expression-first, left-to-right in `AddExpr`/`MulExpr` sequences.
- `let x = Expr;` evaluates `Expr`, binds immutable `x` in the current scope.
- `print Expr;` evaluates `Expr`, writes its value to the host output.
- Arithmetic follows host integer semantics (define overflow behavior: e.g., wrap as in the target integer type or signal UB—TBD). Division by zero: runtime error or UB—TBD.

## IR Mapping (source → Zenith dialect)
- Literals (`Number`, `String`) → `zenith.constant` with an attribute payload and result type.
- Identifier references → `mlir::Value` lookups during HIR→MLIR lowering (no dedicated op if SSA is threaded directly).
- `+` → `zenith.add`, `-` → `zenith.sub`, `*` → `zenith.mul`, `/` → `zenith.div` on compatible numeric types.
- `print` → `zenith.print` consuming the evaluated value.
- Scoping/lets are handled by SSA value binding in the lowering; no dedicated `let` op unless added later.

## Consistency and Evolution
- When grammar changes: update ANTLR files and this EBNF; add parser tests.
- When semantics change: update static/dynamic sections and MLIR op verifiers/lowering.
- When IR surface changes: update `include/zenith/td/ZenithOps.td` and the mapping above.

## Open Points (fill in as you decide)
- Numeric tower and overflow rules.
- Bool type and comparison ops (add productions and ops).
- Functions/blocks/control flow (extend grammar and IR mapping).
- Error handling policy (diagnostics vs. UB).

