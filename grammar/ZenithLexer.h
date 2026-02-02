
#include <deque>
#include <vector>
#include <string>


// Generated from ../../grammar/ZenithLexer.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  ZenithLexer : public antlr4::Lexer {
public:
  enum {
    INDENT = 1, DEDENT = 2, IF = 3, ELSE = 4, FOR = 5, WHILE = 6, RETURN = 7, 
    IN = 8, PRINT = 9, LET = 10, TRUE = 11, FALSE = 12, NULL_ = 13, NOT_WORD = 14, 
    PLUS = 15, MINUS = 16, STAR = 17, DIV = 18, MOD = 19, POW = 20, EQ = 21, 
    NEQ = 22, LT = 23, LE = 24, GT = 25, GE = 26, AND = 27, OR = 28, NOT = 29, 
    AMPERSAND = 30, PIPE = 31, CARET = 32, TILDE = 33, LSHIFT = 34, RSHIFT = 35, 
    LPAREN = 36, RPAREN = 37, LBRACE = 38, RBRACE = 39, LBRACKET = 40, RBRACKET = 41, 
    SEMICOLON = 42, COMMA = 43, DOTDOT = 44, DOT = 45, COLON = 46, ARROW = 47, 
    EQUALS = 48, INTEGER = 49, FLOAT = 50, STRING = 51, IDENTIFIER = 52, 
    NEWLINE = 53, WS = 54, COMMENT = 55, BLOCK_COMMENT = 56
  };

  explicit ZenithLexer(antlr4::CharStream *input);

  ~ZenithLexer() override;


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


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

