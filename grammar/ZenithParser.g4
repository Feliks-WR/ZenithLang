parser grammar ZenithParser;
options { tokenVocab=ZenithLexer; }

program: (stmt | NEWLINE)* EOF;

stmt: simple_stmt (SEMI)?;

simple_stmt
     : assignment
     | procedure_declaration
     | subroutine_declaration
     | function_declaration
     | fn_declaration
     | procedure_call
     | function_call
     | UNSAFE LBRACE (stmt | NEWLINE)* RBRACE
     ;

assignment: ID (COLON type)? ASSIGN expr;

type: base_type
    | base_type range_spec
    | base_type dependent_constraints
    | dependent_type
    | union_type
    | array_type
    | TYPE_KW LPAREN expr RPAREN
    ;

base_type: INT_KW
         | STR_KW
         | FLOAT_KW
         | BOOL_KW
         ;

union_type: base_type (PIPE base_type)+;

array_type: LBRACK array_element_type (SEMI array_size)? RBRACK;

array_element_type: type;

array_size: expr;

range_spec: (LBRACK | LPAREN) expr RANGE expr ( RBRACK | RPAREN );

// Dependent type with explicit variable: int{n : 0..100, constraint1, constraint2}
dependent_type: base_type LBRACE ID (COLON constraint_list)? RBRACE;

// Inline constraints without explicit variable: int @ sorted, int @ notNaN
dependent_constraints: AT constraint_list;

// Comma-separated list of constraints (either ranges or identifiers from stdlib)
constraint_list: constraint (COMMA constraint)*;

// A constraint can be a range or a named constraint from stdlib
constraint: ID                    // constraint from stdlib: sorted, even, notNaN, etc
          | expr RANGE expr       // inline range: 0..100
          | LPAREN expr RANGE expr RPAREN
          ;

expr
    : concat_expr
    ;

concat_expr
    : add_expr (CONCAT add_expr)*
    ;

add_expr
    : mul_expr ((PLUS | MINUS) mul_expr)*
    ;

mul_expr
    : power_expr ((STAR | SLASH) power_expr)*
    ;

power_expr
    : primary (POWER primary)*
    ;

primary
    : INT
    | FLOAT
    | BOOL
    | STRING
    | ID
    | array
    | function_call
    | UNSAFE LBRACE expr RBRACE
    | LPAREN expr RPAREN
    ;

array: LBRACK (expr (COMMA expr)*)? RBRACK;

purity_spec
    : MATH    // full purity, no exceptions etc either, total
    | PURE    // storng purity
    | FUNC;   // weak purity, no side effects but can have signals etc
// features like exceptions may be added later

procedure_declaration: UNSAFE? PROC ID LPAREN (parameter (COMMA parameter)*)? RPAREN (ARROW type)?
function_body;

function_declaration: UNSAFE? purity_spec ID LPAREN (parameter (COMMA parameter)*)? RPAREN (ARROW type)?
function_body;

subroutine_declaration: UNSAFE? SUBROUTINE ID LPAREN (parameter (COMMA parameter)*)? RPAREN (ARROW type)?
function_body;

// fn declarations support multiple syntaxes and optional effect annotations
fn_declaration
    : UNSAFE? FN ID LPAREN (parameter (COMMA parameter)*)? RPAREN (ARROW type)? function_body  // traditional: fn name(params) -> ret { ... }
    | UNSAFE? FN ID COLON function_type_signature                                              // type signature: fn name : int -> int
    | UNSAFE? FN ID LPAREN (parameter (COMMA parameter)*)? RPAREN ASSIGN expr                 // lambda: fn name (x) = x**2
    ;

// Function type signature for declarations: type -> type (supports effects/monads)
function_type_signature
    : type ARROW type (PIPE effect_list)?  // type -> type | IO, Exception
    ;

// Effect/monad annotations for fn functions
effect_list
    : ID (COMMA ID)*
    ;

function_body
    : LBRACE (stmt | NEWLINE)* RBRACE
    | ASSIGN expr
    ;

procedure_call: CALL ID (expr (COMMA expr)*)?;
function_call: ID ((expr (COMMA expr)*) | (LPAREN (expr (COMMA expr)*)? RPAREN))?;

parameter: ID (COLON type)?;
