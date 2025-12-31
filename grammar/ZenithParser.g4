parser grammar ZenithParser;
options { tokenVocab=ZenithLexer; }

program: (stmt | NEWLINE)* EOF;

stmt: simple_stmt (SEMI)?;

simple_stmt
     : assignment
     | procedure_declaration
     | subroutine_declaration
     | function_declaration
     | procedure_call
     | function_call
     | UNSAFE LBRACE (stmt | NEWLINE)* RBRACE
     ;

assignment: ID (COLON type)? ASSIGN expr;

type: base_type
    | base_type range_spec
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

array_type: LBRACK (union_type | base_type) (SEMI INT)? RBRACK;

range_spec: (LBRACK | LPAREN) INT RANGE INT ( RBRACK | RPAREN );

expr
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
    | FUNC;   // weak purity, no side effects but can have exceptions and signals etc
// features like exceptions may be added later

procedure_declaration: UNSAFE? PROC ID LPAREN (parameter (COMMA parameter)*)? RPAREN (ARROW type)?
function_body;

function_declaration: UNSAFE? purity_spec ID LPAREN (parameter (COMMA parameter)*)? RPAREN (ARROW type)?
function_body;

subroutine_declaration: UNSAFE? SUBROUTINE ID LPAREN (parameter (COMMA parameter)*)? RPAREN (ARROW type)?
function_body;

function_body
    : LBRACE (stmt | NEWLINE)* RBRACE
    | ASSIGN expr
    ;

procedure_call: CALL ID (expr (COMMA expr)*)?;
function_call: ID ((expr (COMMA expr)*) | (LPAREN (expr (COMMA expr)*)? RPAREN))?;

parameter: ID (COLON type)?;
