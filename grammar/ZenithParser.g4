// ANTLR4 Parser for Zenith
parser grammar ZenithParser;

options { tokenVocab=ZenithLexer; }

program: NEWLINE* (statement (semi statement)* semi?)? NEWLINE* EOF;
statement: varDeclaration | functionDecl | equation | ifStatement | whileStatement | forStatement 
        | returnStatement | printStatement | exprStatement | blockStatement;

semi: (SEMICOLON | NEWLINE)+;

// Forward declarations: x, y : real
varDeclaration: identifierList (COLON type)?;
identifierList: IDENTIFIER (COMMA IDENTIFIER)*;

// Function declarations
functionDecl:
        IDENTIFIER LPAREN parameterList? RPAREN (ARROW type)?
        (
                EQUALS expression
            | EQUALS? NEWLINE* blockStatement
        );

parameterList: parameter (COMMA parameter)*;
parameter: IDENTIFIER (COLON type)?;

// Equations: x + y = 3; or x = 5;
equation: expression EQUALS expression;

// Expression statement (for function calls like: println x, y)
exprStatement: expression;

// Blocks (for function bodies and if/else)
blockStatement:
                        LBRACE NEWLINE* (statement (semi statement)* semi?)? NEWLINE* RBRACE
                | NEWLINE INDENT (statement (semi statement)* semi?)? DEDENT
                ;

ifStatement: IF expression (NEWLINE INDENT (statement (semi statement)* semi?)? DEDENT | blockStatement) (NEWLINE* ELSE (NEWLINE INDENT (statement (semi statement)* semi?)? DEDENT | blockStatement))?;
whileStatement: WHILE expression (NEWLINE INDENT (statement (semi statement)* semi?)? DEDENT | blockStatement);
forStatement: FOR IDENTIFIER IN expression (NEWLINE INDENT (statement (semi statement)* semi?)? DEDENT | blockStatement);
returnStatement: RETURN expression?;
printStatement: PRINT expression (COMMA expression)*;

// Types
type: IDENTIFIER (LBRACKET INTEGER RBRACKET)* | AMPERSAND type;

// Expressions with function application: f x, y
expression: logicalOrExpr;
logicalOrExpr: logicalAndExpr (OR logicalAndExpr)*;
logicalAndExpr: bitwiseOrExpr (AND bitwiseOrExpr)*;
bitwiseOrExpr: bitwiseXorExpr (PIPE bitwiseXorExpr)*;
bitwiseXorExpr: bitwiseAndExpr (CARET bitwiseAndExpr)*;
bitwiseAndExpr: equalityExpr (AMPERSAND equalityExpr)*;
equalityExpr: relationalExpr ((EQ | NEQ) relationalExpr)*;
relationalExpr: shiftExpr ((LT | LE | GT | GE) shiftExpr)*;
shiftExpr: additiveExpr ((LSHIFT | RSHIFT) additiveExpr)*;
additiveExpr: multiplicativeExpr ((PLUS | MINUS) multiplicativeExpr)*;
multiplicativeExpr: powerExpr ((STAR | DIV | MOD) powerExpr)*;
// Make power (`**`) right-associative: unary ** power
powerExpr: unaryExpr (POW powerExpr)?;
unaryExpr: (NOT | MINUS | TILDE | AMPERSAND) unaryExpr | callExpr;

// Function application: f x y or f(x, y) or arr[i] or obj.field
callExpr: primaryExpr (callSuffix)*;
callSuffix: 
    LBRACKET expression RBRACKET                    // array indexing: arr[0]
    | DOT IDENTIFIER                                // member access: obj.field
    | LPAREN (expression (COMMA expression)*)? RPAREN // function call: f(x, y)
        | primaryExpr (COMMA primaryExpr)*              // function application: f x, y, z
    ;

primaryExpr: INTEGER | FLOAT | STRING | TRUE | FALSE | IDENTIFIER | LPAREN expression RPAREN;
