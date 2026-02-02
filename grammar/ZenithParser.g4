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

<<<<<<< HEAD
// Types with dependent type support
type: dependentType;

dependentType: baseType constraint?;

baseType: 
    IDENTIFIER                           // basic types: int, float, bool, etc
    | pointerType                        // pointer: *T
    | arrayType                          // array: [T; N]
    ;

pointerType: STAR baseType;

arrayType: LBRACKET baseType SEMICOLON IDENTIFIER RBRACKET;

// Constraints: {it != 0}, {(!=0)}, {nonnull}, {1..10}
constraint: LBRACE constraintExpr RBRACE;

constraintExpr:
    IDENTIFIER EQ INTEGER                // single value: 5
    | INTEGER DOTDOT INTEGER             // range: 1..10
    | predicateExpr                      // predicate: it != 0 or (!=0)
    | IDENTIFIER                         // named constraint: nonnull
    ;

predicateExpr:
    LPAREN comparisonOp INTEGER RPAREN   // implicit: (!=0)
    | IT comparisonOp INTEGER            // explicit: it != 0
    ;

comparisonOp: EQ | NEQ | LT | LE | GT | GE;
=======
// Types with optional dependent type predicates
type: baseType dependentPredicate?;
baseType: innerType (LBRACKET INTEGER RBRACKET)* | AMPERSAND baseType;
innerType: LBRACKET baseType RBRACKET | IDENTIFIER;
dependentPredicate: LBRACE predicate RBRACE;
// Dependent type predicates
// Examples: {!= 0}, {not blank}, {1..10}, {sorted}, {n > 0}, {42}, etc.
predicate: rangePredicate | infixPredicate | unaryPredicate | complexPredicate | predicateValue;

rangePredicate: predicateValue DOTDOT predicateValue;

// Infix predicates: left operand is implicit (comparison with the value itself)
// Example: {!= 0} means "not equal to 0"
infixPredicate: predicateOp predicateValue;

// Unary: -x, not x (for things like {-1}, {not something})
unaryPredicate: (NOT | NOT_WORD | MINUS) predicateValue;

// Complex: sorted, blank, custom_check(x)
complexPredicate: IDENTIFIER (LPAREN predicateArgList? RPAREN)?;

predicateArgList: predicate (COMMA predicate)*;

// Predicate values (literals and identifiers, no expressions to avoid left recursion)
predicateValue: IDENTIFIER | INTEGER | FLOAT | STRING | LPAREN predicate RPAREN;

predicateOp: EQ | NEQ | LT | LE | GT | GE | PLUS | MINUS | STAR | DIV | MOD | AND | OR;
>>>>>>> f44d684 (Add initial implementation of Zenith parser and visitor classes)

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
    ;

primaryExpr: INTEGER | FLOAT | STRING | TRUE | FALSE | IDENTIFIER | LPAREN expression RPAREN;
