// ANTLR4 Lexer for Zenith
lexer grammar ZenithLexer;

// Declare indentation tokens (will be emitted from lexer actions)
tokens { INDENT, DEDENT }

@header {
#include <deque>
#include <vector>
#include <string>
}

@members {
	std::deque<std::unique_ptr<antlr4::Token>> tokens_queue;
	std::vector<int> indents = {0};

	int getIndentationCount(const std::string &s) {
		int count = 0;
		for (char ch : s) {
			if (ch == '\t') count += 8 - (count % 8);
			else count += 1;
		}
		return count;
	}

	std::unique_ptr<antlr4::Token> nextToken() override {
		if (!tokens_queue.empty()) {
			auto t = std::move(tokens_queue.front()); tokens_queue.pop_front();
			return t;
		}

		auto next = antlr4::Lexer::nextToken();

		if (next && next->getType() == NEWLINE) {
			// Count spaces/tabs at start of next line (lookahead without consuming beyond spaces)
			int la = _input->LA(1);
			std::string spaces;
			while (la == ' ' || la == '\t') {
				spaces.push_back((char)la);
				_input->consume();
				la = _input->LA(1);
			}

			// If the line is empty (next is newline or EOF), just return the NEWLINE token
			if (la == '\r' || la == '\n' || la == antlr4::Token::EOF) {
				return next;
			}

			int indent = getIndentationCount(spaces);
			int prev = indents.back();

			if (indent > prev) {
				indents.push_back(indent);
				tokens_queue.push_back(std::move(next)); // first emit the NEWLINE
				tokens_queue.push_back(std::make_unique<antlr4::CommonToken>(INDENT, ""));
				return this->nextToken();
			}

			if (indent < prev) {
				tokens_queue.push_back(std::move(next));
				while (indent < indents.back()) {
					indents.pop_back();
					tokens_queue.push_back(std::make_unique<antlr4::CommonToken>(DEDENT, ""));
				}
				return this->nextToken();
			}

			// same indent as before
			return next;
		}

		if (!next || next->getType() == antlr4::Token::EOF) {
			// Emit DEDENTs for any remaining indents
			while (indents.size() > 1) {
				indents.pop_back();
				tokens_queue.push_back(std::make_unique<antlr4::CommonToken>(DEDENT, ""));
			}
			tokens_queue.push_back(std::move(next));
			return this->nextToken();
		}

		return next;
	}
}

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
