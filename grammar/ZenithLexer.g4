// ANTLR4 Lexer for Zenith
lexer grammar ZenithLexer;

FN: 'fn';
IF: 'if';
ELSE: 'else';
FOR: 'for';
WHILE: 'while';
RETURN: 'return';
IN: 'in';
TRUE: 'true';
FALSE: 'false';

PLUS: '+';
MINUS: '-';
STAR: '*';
DIV: '/';
MOD: '%';
POW: '**';
EQ: '==';
NEQ: '!=';
LT: '<';
LE: '<=';
GT: '>';
GE: '>=';
AND: '&&';
OR: '||';
NOT: '!';
AMPERSAND: '&';
PIPE: '|';
CARET: '^';
TILDE: '~';
LSHIFT: '<<';
RSHIFT: '>>';

LPAREN: '(';
RPAREN: ')';
LBRACE: '{';
RBRACE: '}';
LBRACKET: '[';
RBRACKET: ']';
SEMICOLON: ';';
COMMA: ',';
DOT: '.';
COLON: ':';
ARROW: '->';
EQUALS: '=';

INTEGER: [0-9]+;
FLOAT: [0-9]+ '.' [0-9]+;
STRING: '"' (~["\\\r\n] | '\\' .)* '"';
IDENTIFIER: [a-zA-Z_][a-zA-Z0-9_]*;

NEWLINE: ('\r'? '\n' | '\r')+;
WS: [ \t\f]+ -> channel(HIDDEN);
COMMENT: '//' ~[\r\n]* -> channel(HIDDEN);
BLOCK_COMMENT: '/*' .*? '*/' -> channel(HIDDEN);
