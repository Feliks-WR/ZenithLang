
// Generated from ../../grammar/ZenithParser.g4 by ANTLR 4.13.2


#include "ZenithParserVisitor.h"

#include "ZenithParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct ZenithParserStaticData final {
  ZenithParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  ZenithParserStaticData(const ZenithParserStaticData&) = delete;
  ZenithParserStaticData(ZenithParserStaticData&&) = delete;
  ZenithParserStaticData& operator=(const ZenithParserStaticData&) = delete;
  ZenithParserStaticData& operator=(ZenithParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag zenithparserParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<ZenithParserStaticData> zenithparserParserStaticData = nullptr;

void zenithparserParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (zenithparserParserStaticData != nullptr) {
    return;
  }
#else
  assert(zenithparserParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<ZenithParserStaticData>(
    std::vector<std::string>{
      "program", "statement", "semi", "varDeclaration", "identifierList", 
      "functionDecl", "parameterList", "parameter", "equation", "exprStatement", 
      "blockStatement", "ifStatement", "whileStatement", "forStatement", 
      "returnStatement", "printStatement", "type", "baseType", "dependentPredicate", 
      "predicate", "rangePredicate", "unaryPredicate", "binaryPredicate", 
      "complexPredicate", "predicateArgList", "predicateValue", "predicateOp", 
      "expression", "logicalOrExpr", "logicalAndExpr", "bitwiseOrExpr", 
      "bitwiseXorExpr", "bitwiseAndExpr", "equalityExpr", "relationalExpr", 
      "shiftExpr", "additiveExpr", "multiplicativeExpr", "powerExpr", "unaryExpr", 
      "callExpr", "callSuffix", "primaryExpr"
    },
    std::vector<std::string>{
      "", "", "", "'if'", "'else'", "'for'", "'while'", "'return'", "'in'", 
      "'print'", "'let'", "'true'", "'false'", "'null'", "'not'", "'+'", 
      "'-'", "'*'", "'/'", "'%'", "'**'", "'=='", "'!='", "'<'", "'<='", 
      "'>'", "'>='", "'&&'", "'||'", "'!'", "'&'", "'|'", "'^'", "'~'", 
      "'<<'", "'>>'", "'('", "')'", "'{'", "'}'", "'['", "']'", "';'", "','", 
      "'..'", "'.'", "':'", "'->'", "'='"
    },
    std::vector<std::string>{
      "", "INDENT", "DEDENT", "IF", "ELSE", "FOR", "WHILE", "RETURN", "IN", 
      "PRINT", "LET", "TRUE", "FALSE", "NULL", "NOT_WORD", "PLUS", "MINUS", 
      "STAR", "DIV", "MOD", "POW", "EQ", "NEQ", "LT", "LE", "GT", "GE", 
      "AND", "OR", "NOT", "AMPERSAND", "PIPE", "CARET", "TILDE", "LSHIFT", 
      "RSHIFT", "LPAREN", "RPAREN", "LBRACE", "RBRACE", "LBRACKET", "RBRACKET", 
      "SEMICOLON", "COMMA", "DOTDOT", "DOT", "COLON", "ARROW", "EQUALS", 
      "INTEGER", "FLOAT", "STRING", "IDENTIFIER", "NEWLINE", "WS", "COMMENT", 
      "BLOCK_COMMENT"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,56,547,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,7,20,2,21,7,
  	21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,2,27,7,27,2,28,7,
  	28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,7,33,2,34,7,34,2,35,7,
  	35,2,36,7,36,2,37,7,37,2,38,7,38,2,39,7,39,2,40,7,40,2,41,7,41,2,42,7,
  	42,1,0,5,0,88,8,0,10,0,12,0,91,9,0,1,0,1,0,4,0,95,8,0,11,0,12,0,96,1,
  	0,1,0,5,0,101,8,0,10,0,12,0,104,9,0,1,0,3,0,107,8,0,3,0,109,8,0,1,0,5,
  	0,112,8,0,10,0,12,0,115,9,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  	1,1,1,3,1,129,8,1,1,2,1,2,1,3,1,3,1,3,3,3,136,8,3,1,4,1,4,1,4,5,4,141,
  	8,4,10,4,12,4,144,9,4,1,5,1,5,1,5,3,5,149,8,5,1,5,1,5,1,5,3,5,154,8,5,
  	1,5,1,5,1,5,3,5,159,8,5,1,5,5,5,162,8,5,10,5,12,5,165,9,5,1,5,3,5,168,
  	8,5,1,6,1,6,1,6,5,6,173,8,6,10,6,12,6,176,9,6,1,7,1,7,1,7,3,7,181,8,7,
  	1,8,1,8,1,8,1,8,1,9,1,9,1,10,1,10,5,10,191,8,10,10,10,12,10,194,9,10,
  	1,10,1,10,1,10,1,10,5,10,200,8,10,10,10,12,10,203,9,10,1,10,3,10,206,
  	8,10,3,10,208,8,10,1,10,5,10,211,8,10,10,10,12,10,214,9,10,1,10,1,10,
  	1,10,1,10,1,10,1,10,1,10,5,10,223,8,10,10,10,12,10,226,9,10,1,10,3,10,
  	229,8,10,3,10,231,8,10,1,10,3,10,234,8,10,1,11,1,11,1,11,1,11,1,11,1,
  	11,1,11,1,11,5,11,244,8,11,10,11,12,11,247,9,11,1,11,3,11,250,8,11,3,
  	11,252,8,11,1,11,1,11,3,11,256,8,11,1,11,5,11,259,8,11,10,11,12,11,262,
  	9,11,1,11,1,11,1,11,1,11,1,11,1,11,1,11,5,11,271,8,11,10,11,12,11,274,
  	9,11,1,11,3,11,277,8,11,3,11,279,8,11,1,11,1,11,3,11,283,8,11,3,11,285,
  	8,11,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,5,12,295,8,12,10,12,12,12,
  	298,9,12,1,12,3,12,301,8,12,3,12,303,8,12,1,12,1,12,3,12,307,8,12,1,13,
  	1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,5,13,319,8,13,10,13,12,13,
  	322,9,13,1,13,3,13,325,8,13,3,13,327,8,13,1,13,1,13,3,13,331,8,13,1,14,
  	1,14,3,14,335,8,14,1,15,1,15,1,15,1,15,5,15,341,8,15,10,15,12,15,344,
  	9,15,1,16,1,16,3,16,348,8,16,1,17,1,17,1,17,1,17,5,17,354,8,17,10,17,
  	12,17,357,9,17,1,17,1,17,3,17,361,8,17,1,18,1,18,1,18,1,18,1,19,1,19,
  	1,19,1,19,3,19,371,8,19,1,20,1,20,1,20,1,20,1,21,1,21,1,21,1,22,1,22,
  	1,22,1,22,4,22,384,8,22,11,22,12,22,385,1,23,1,23,1,23,3,23,391,8,23,
  	1,23,3,23,394,8,23,1,24,1,24,1,24,5,24,399,8,24,10,24,12,24,402,9,24,
  	1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,25,3,25,412,8,25,1,26,1,26,1,27,
  	1,27,1,28,1,28,1,28,5,28,421,8,28,10,28,12,28,424,9,28,1,29,1,29,1,29,
  	5,29,429,8,29,10,29,12,29,432,9,29,1,30,1,30,1,30,5,30,437,8,30,10,30,
  	12,30,440,9,30,1,31,1,31,1,31,5,31,445,8,31,10,31,12,31,448,9,31,1,32,
  	1,32,1,32,5,32,453,8,32,10,32,12,32,456,9,32,1,33,1,33,1,33,5,33,461,
  	8,33,10,33,12,33,464,9,33,1,34,1,34,1,34,5,34,469,8,34,10,34,12,34,472,
  	9,34,1,35,1,35,1,35,5,35,477,8,35,10,35,12,35,480,9,35,1,36,1,36,1,36,
  	5,36,485,8,36,10,36,12,36,488,9,36,1,37,1,37,1,37,5,37,493,8,37,10,37,
  	12,37,496,9,37,1,38,1,38,1,38,3,38,501,8,38,1,39,1,39,1,39,3,39,506,8,
  	39,1,40,1,40,5,40,510,8,40,10,40,12,40,513,9,40,1,41,1,41,1,41,1,41,1,
  	41,1,41,1,41,1,41,1,41,1,41,5,41,525,8,41,10,41,12,41,528,9,41,3,41,530,
  	8,41,1,41,3,41,533,8,41,1,42,1,42,1,42,1,42,1,42,1,42,1,42,1,42,1,42,
  	1,42,3,42,545,8,42,1,42,0,0,43,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,
  	30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,
  	76,78,80,82,84,0,8,3,0,14,14,16,16,29,29,2,0,15,19,21,28,1,0,21,22,1,
  	0,23,26,1,0,34,35,1,0,15,16,1,0,17,19,3,0,16,16,29,30,33,33,593,0,89,
  	1,0,0,0,2,128,1,0,0,0,4,130,1,0,0,0,6,132,1,0,0,0,8,137,1,0,0,0,10,145,
  	1,0,0,0,12,169,1,0,0,0,14,177,1,0,0,0,16,182,1,0,0,0,18,186,1,0,0,0,20,
  	233,1,0,0,0,22,235,1,0,0,0,24,286,1,0,0,0,26,308,1,0,0,0,28,332,1,0,0,
  	0,30,336,1,0,0,0,32,345,1,0,0,0,34,360,1,0,0,0,36,362,1,0,0,0,38,370,
  	1,0,0,0,40,372,1,0,0,0,42,376,1,0,0,0,44,379,1,0,0,0,46,387,1,0,0,0,48,
  	395,1,0,0,0,50,411,1,0,0,0,52,413,1,0,0,0,54,415,1,0,0,0,56,417,1,0,0,
  	0,58,425,1,0,0,0,60,433,1,0,0,0,62,441,1,0,0,0,64,449,1,0,0,0,66,457,
  	1,0,0,0,68,465,1,0,0,0,70,473,1,0,0,0,72,481,1,0,0,0,74,489,1,0,0,0,76,
  	497,1,0,0,0,78,505,1,0,0,0,80,507,1,0,0,0,82,532,1,0,0,0,84,544,1,0,0,
  	0,86,88,5,53,0,0,87,86,1,0,0,0,88,91,1,0,0,0,89,87,1,0,0,0,89,90,1,0,
  	0,0,90,108,1,0,0,0,91,89,1,0,0,0,92,102,3,2,1,0,93,95,3,4,2,0,94,93,1,
  	0,0,0,95,96,1,0,0,0,96,94,1,0,0,0,96,97,1,0,0,0,97,98,1,0,0,0,98,99,3,
  	2,1,0,99,101,1,0,0,0,100,94,1,0,0,0,101,104,1,0,0,0,102,100,1,0,0,0,102,
  	103,1,0,0,0,103,106,1,0,0,0,104,102,1,0,0,0,105,107,3,4,2,0,106,105,1,
  	0,0,0,106,107,1,0,0,0,107,109,1,0,0,0,108,92,1,0,0,0,108,109,1,0,0,0,
  	109,113,1,0,0,0,110,112,5,53,0,0,111,110,1,0,0,0,112,115,1,0,0,0,113,
  	111,1,0,0,0,113,114,1,0,0,0,114,116,1,0,0,0,115,113,1,0,0,0,116,117,5,
  	0,0,1,117,1,1,0,0,0,118,129,3,6,3,0,119,129,3,10,5,0,120,129,3,16,8,0,
  	121,129,3,22,11,0,122,129,3,24,12,0,123,129,3,26,13,0,124,129,3,28,14,
  	0,125,129,3,30,15,0,126,129,3,18,9,0,127,129,3,20,10,0,128,118,1,0,0,
  	0,128,119,1,0,0,0,128,120,1,0,0,0,128,121,1,0,0,0,128,122,1,0,0,0,128,
  	123,1,0,0,0,128,124,1,0,0,0,128,125,1,0,0,0,128,126,1,0,0,0,128,127,1,
  	0,0,0,129,3,1,0,0,0,130,131,5,42,0,0,131,5,1,0,0,0,132,135,3,8,4,0,133,
  	134,5,46,0,0,134,136,3,32,16,0,135,133,1,0,0,0,135,136,1,0,0,0,136,7,
  	1,0,0,0,137,142,5,52,0,0,138,139,5,43,0,0,139,141,5,52,0,0,140,138,1,
  	0,0,0,141,144,1,0,0,0,142,140,1,0,0,0,142,143,1,0,0,0,143,9,1,0,0,0,144,
  	142,1,0,0,0,145,146,5,52,0,0,146,148,5,36,0,0,147,149,3,12,6,0,148,147,
  	1,0,0,0,148,149,1,0,0,0,149,150,1,0,0,0,150,153,5,37,0,0,151,152,5,47,
  	0,0,152,154,3,32,16,0,153,151,1,0,0,0,153,154,1,0,0,0,154,167,1,0,0,0,
  	155,156,5,48,0,0,156,168,3,54,27,0,157,159,5,48,0,0,158,157,1,0,0,0,158,
  	159,1,0,0,0,159,163,1,0,0,0,160,162,5,53,0,0,161,160,1,0,0,0,162,165,
  	1,0,0,0,163,161,1,0,0,0,163,164,1,0,0,0,164,166,1,0,0,0,165,163,1,0,0,
  	0,166,168,3,20,10,0,167,155,1,0,0,0,167,158,1,0,0,0,168,11,1,0,0,0,169,
  	174,3,14,7,0,170,171,5,43,0,0,171,173,3,14,7,0,172,170,1,0,0,0,173,176,
  	1,0,0,0,174,172,1,0,0,0,174,175,1,0,0,0,175,13,1,0,0,0,176,174,1,0,0,
  	0,177,180,5,52,0,0,178,179,5,46,0,0,179,181,3,32,16,0,180,178,1,0,0,0,
  	180,181,1,0,0,0,181,15,1,0,0,0,182,183,3,54,27,0,183,184,5,48,0,0,184,
  	185,3,54,27,0,185,17,1,0,0,0,186,187,3,54,27,0,187,19,1,0,0,0,188,192,
  	5,38,0,0,189,191,5,53,0,0,190,189,1,0,0,0,191,194,1,0,0,0,192,190,1,0,
  	0,0,192,193,1,0,0,0,193,207,1,0,0,0,194,192,1,0,0,0,195,201,3,2,1,0,196,
  	197,3,4,2,0,197,198,3,2,1,0,198,200,1,0,0,0,199,196,1,0,0,0,200,203,1,
  	0,0,0,201,199,1,0,0,0,201,202,1,0,0,0,202,205,1,0,0,0,203,201,1,0,0,0,
  	204,206,3,4,2,0,205,204,1,0,0,0,205,206,1,0,0,0,206,208,1,0,0,0,207,195,
  	1,0,0,0,207,208,1,0,0,0,208,212,1,0,0,0,209,211,5,53,0,0,210,209,1,0,
  	0,0,211,214,1,0,0,0,212,210,1,0,0,0,212,213,1,0,0,0,213,215,1,0,0,0,214,
  	212,1,0,0,0,215,234,5,39,0,0,216,217,5,53,0,0,217,230,5,1,0,0,218,224,
  	3,2,1,0,219,220,3,4,2,0,220,221,3,2,1,0,221,223,1,0,0,0,222,219,1,0,0,
  	0,223,226,1,0,0,0,224,222,1,0,0,0,224,225,1,0,0,0,225,228,1,0,0,0,226,
  	224,1,0,0,0,227,229,3,4,2,0,228,227,1,0,0,0,228,229,1,0,0,0,229,231,1,
  	0,0,0,230,218,1,0,0,0,230,231,1,0,0,0,231,232,1,0,0,0,232,234,5,2,0,0,
  	233,188,1,0,0,0,233,216,1,0,0,0,234,21,1,0,0,0,235,236,5,3,0,0,236,255,
  	3,54,27,0,237,238,5,53,0,0,238,251,5,1,0,0,239,245,3,2,1,0,240,241,3,
  	4,2,0,241,242,3,2,1,0,242,244,1,0,0,0,243,240,1,0,0,0,244,247,1,0,0,0,
  	245,243,1,0,0,0,245,246,1,0,0,0,246,249,1,0,0,0,247,245,1,0,0,0,248,250,
  	3,4,2,0,249,248,1,0,0,0,249,250,1,0,0,0,250,252,1,0,0,0,251,239,1,0,0,
  	0,251,252,1,0,0,0,252,253,1,0,0,0,253,256,5,2,0,0,254,256,3,20,10,0,255,
  	237,1,0,0,0,255,254,1,0,0,0,256,284,1,0,0,0,257,259,5,53,0,0,258,257,
  	1,0,0,0,259,262,1,0,0,0,260,258,1,0,0,0,260,261,1,0,0,0,261,263,1,0,0,
  	0,262,260,1,0,0,0,263,282,5,4,0,0,264,265,5,53,0,0,265,278,5,1,0,0,266,
  	272,3,2,1,0,267,268,3,4,2,0,268,269,3,2,1,0,269,271,1,0,0,0,270,267,1,
  	0,0,0,271,274,1,0,0,0,272,270,1,0,0,0,272,273,1,0,0,0,273,276,1,0,0,0,
  	274,272,1,0,0,0,275,277,3,4,2,0,276,275,1,0,0,0,276,277,1,0,0,0,277,279,
  	1,0,0,0,278,266,1,0,0,0,278,279,1,0,0,0,279,280,1,0,0,0,280,283,5,2,0,
  	0,281,283,3,20,10,0,282,264,1,0,0,0,282,281,1,0,0,0,283,285,1,0,0,0,284,
  	260,1,0,0,0,284,285,1,0,0,0,285,23,1,0,0,0,286,287,5,6,0,0,287,306,3,
  	54,27,0,288,289,5,53,0,0,289,302,5,1,0,0,290,296,3,2,1,0,291,292,3,4,
  	2,0,292,293,3,2,1,0,293,295,1,0,0,0,294,291,1,0,0,0,295,298,1,0,0,0,296,
  	294,1,0,0,0,296,297,1,0,0,0,297,300,1,0,0,0,298,296,1,0,0,0,299,301,3,
  	4,2,0,300,299,1,0,0,0,300,301,1,0,0,0,301,303,1,0,0,0,302,290,1,0,0,0,
  	302,303,1,0,0,0,303,304,1,0,0,0,304,307,5,2,0,0,305,307,3,20,10,0,306,
  	288,1,0,0,0,306,305,1,0,0,0,307,25,1,0,0,0,308,309,5,5,0,0,309,310,5,
  	52,0,0,310,311,5,8,0,0,311,330,3,54,27,0,312,313,5,53,0,0,313,326,5,1,
  	0,0,314,320,3,2,1,0,315,316,3,4,2,0,316,317,3,2,1,0,317,319,1,0,0,0,318,
  	315,1,0,0,0,319,322,1,0,0,0,320,318,1,0,0,0,320,321,1,0,0,0,321,324,1,
  	0,0,0,322,320,1,0,0,0,323,325,3,4,2,0,324,323,1,0,0,0,324,325,1,0,0,0,
  	325,327,1,0,0,0,326,314,1,0,0,0,326,327,1,0,0,0,327,328,1,0,0,0,328,331,
  	5,2,0,0,329,331,3,20,10,0,330,312,1,0,0,0,330,329,1,0,0,0,331,27,1,0,
  	0,0,332,334,5,7,0,0,333,335,3,54,27,0,334,333,1,0,0,0,334,335,1,0,0,0,
  	335,29,1,0,0,0,336,337,5,9,0,0,337,342,3,54,27,0,338,339,5,43,0,0,339,
  	341,3,54,27,0,340,338,1,0,0,0,341,344,1,0,0,0,342,340,1,0,0,0,342,343,
  	1,0,0,0,343,31,1,0,0,0,344,342,1,0,0,0,345,347,3,34,17,0,346,348,3,36,
  	18,0,347,346,1,0,0,0,347,348,1,0,0,0,348,33,1,0,0,0,349,355,5,52,0,0,
  	350,351,5,40,0,0,351,352,5,49,0,0,352,354,5,41,0,0,353,350,1,0,0,0,354,
  	357,1,0,0,0,355,353,1,0,0,0,355,356,1,0,0,0,356,361,1,0,0,0,357,355,1,
  	0,0,0,358,359,5,30,0,0,359,361,3,34,17,0,360,349,1,0,0,0,360,358,1,0,
  	0,0,361,35,1,0,0,0,362,363,5,38,0,0,363,364,3,38,19,0,364,365,5,39,0,
  	0,365,37,1,0,0,0,366,371,3,40,20,0,367,371,3,42,21,0,368,371,3,44,22,
  	0,369,371,3,46,23,0,370,366,1,0,0,0,370,367,1,0,0,0,370,368,1,0,0,0,370,
  	369,1,0,0,0,371,39,1,0,0,0,372,373,3,50,25,0,373,374,5,44,0,0,374,375,
  	3,50,25,0,375,41,1,0,0,0,376,377,7,0,0,0,377,378,3,50,25,0,378,43,1,0,
  	0,0,379,383,3,50,25,0,380,381,3,52,26,0,381,382,3,50,25,0,382,384,1,0,
  	0,0,383,380,1,0,0,0,384,385,1,0,0,0,385,383,1,0,0,0,385,386,1,0,0,0,386,
  	45,1,0,0,0,387,393,5,52,0,0,388,390,5,36,0,0,389,391,3,48,24,0,390,389,
  	1,0,0,0,390,391,1,0,0,0,391,392,1,0,0,0,392,394,5,37,0,0,393,388,1,0,
  	0,0,393,394,1,0,0,0,394,47,1,0,0,0,395,400,3,38,19,0,396,397,5,43,0,0,
  	397,399,3,38,19,0,398,396,1,0,0,0,399,402,1,0,0,0,400,398,1,0,0,0,400,
  	401,1,0,0,0,401,49,1,0,0,0,402,400,1,0,0,0,403,412,5,52,0,0,404,412,5,
  	49,0,0,405,412,5,50,0,0,406,412,5,51,0,0,407,408,5,36,0,0,408,409,3,38,
  	19,0,409,410,5,37,0,0,410,412,1,0,0,0,411,403,1,0,0,0,411,404,1,0,0,0,
  	411,405,1,0,0,0,411,406,1,0,0,0,411,407,1,0,0,0,412,51,1,0,0,0,413,414,
  	7,1,0,0,414,53,1,0,0,0,415,416,3,56,28,0,416,55,1,0,0,0,417,422,3,58,
  	29,0,418,419,5,28,0,0,419,421,3,58,29,0,420,418,1,0,0,0,421,424,1,0,0,
  	0,422,420,1,0,0,0,422,423,1,0,0,0,423,57,1,0,0,0,424,422,1,0,0,0,425,
  	430,3,60,30,0,426,427,5,27,0,0,427,429,3,60,30,0,428,426,1,0,0,0,429,
  	432,1,0,0,0,430,428,1,0,0,0,430,431,1,0,0,0,431,59,1,0,0,0,432,430,1,
  	0,0,0,433,438,3,62,31,0,434,435,5,31,0,0,435,437,3,62,31,0,436,434,1,
  	0,0,0,437,440,1,0,0,0,438,436,1,0,0,0,438,439,1,0,0,0,439,61,1,0,0,0,
  	440,438,1,0,0,0,441,446,3,64,32,0,442,443,5,32,0,0,443,445,3,64,32,0,
  	444,442,1,0,0,0,445,448,1,0,0,0,446,444,1,0,0,0,446,447,1,0,0,0,447,63,
  	1,0,0,0,448,446,1,0,0,0,449,454,3,66,33,0,450,451,5,30,0,0,451,453,3,
  	66,33,0,452,450,1,0,0,0,453,456,1,0,0,0,454,452,1,0,0,0,454,455,1,0,0,
  	0,455,65,1,0,0,0,456,454,1,0,0,0,457,462,3,68,34,0,458,459,7,2,0,0,459,
  	461,3,68,34,0,460,458,1,0,0,0,461,464,1,0,0,0,462,460,1,0,0,0,462,463,
  	1,0,0,0,463,67,1,0,0,0,464,462,1,0,0,0,465,470,3,70,35,0,466,467,7,3,
  	0,0,467,469,3,70,35,0,468,466,1,0,0,0,469,472,1,0,0,0,470,468,1,0,0,0,
  	470,471,1,0,0,0,471,69,1,0,0,0,472,470,1,0,0,0,473,478,3,72,36,0,474,
  	475,7,4,0,0,475,477,3,72,36,0,476,474,1,0,0,0,477,480,1,0,0,0,478,476,
  	1,0,0,0,478,479,1,0,0,0,479,71,1,0,0,0,480,478,1,0,0,0,481,486,3,74,37,
  	0,482,483,7,5,0,0,483,485,3,74,37,0,484,482,1,0,0,0,485,488,1,0,0,0,486,
  	484,1,0,0,0,486,487,1,0,0,0,487,73,1,0,0,0,488,486,1,0,0,0,489,494,3,
  	76,38,0,490,491,7,6,0,0,491,493,3,76,38,0,492,490,1,0,0,0,493,496,1,0,
  	0,0,494,492,1,0,0,0,494,495,1,0,0,0,495,75,1,0,0,0,496,494,1,0,0,0,497,
  	500,3,78,39,0,498,499,5,20,0,0,499,501,3,76,38,0,500,498,1,0,0,0,500,
  	501,1,0,0,0,501,77,1,0,0,0,502,503,7,7,0,0,503,506,3,78,39,0,504,506,
  	3,80,40,0,505,502,1,0,0,0,505,504,1,0,0,0,506,79,1,0,0,0,507,511,3,84,
  	42,0,508,510,3,82,41,0,509,508,1,0,0,0,510,513,1,0,0,0,511,509,1,0,0,
  	0,511,512,1,0,0,0,512,81,1,0,0,0,513,511,1,0,0,0,514,515,5,40,0,0,515,
  	516,3,54,27,0,516,517,5,41,0,0,517,533,1,0,0,0,518,519,5,45,0,0,519,533,
  	5,52,0,0,520,529,5,36,0,0,521,526,3,54,27,0,522,523,5,43,0,0,523,525,
  	3,54,27,0,524,522,1,0,0,0,525,528,1,0,0,0,526,524,1,0,0,0,526,527,1,0,
  	0,0,527,530,1,0,0,0,528,526,1,0,0,0,529,521,1,0,0,0,529,530,1,0,0,0,530,
  	531,1,0,0,0,531,533,5,37,0,0,532,514,1,0,0,0,532,518,1,0,0,0,532,520,
  	1,0,0,0,533,83,1,0,0,0,534,545,5,49,0,0,535,545,5,50,0,0,536,545,5,51,
  	0,0,537,545,5,11,0,0,538,545,5,12,0,0,539,545,5,52,0,0,540,541,5,36,0,
  	0,541,542,3,54,27,0,542,543,5,37,0,0,543,545,1,0,0,0,544,534,1,0,0,0,
  	544,535,1,0,0,0,544,536,1,0,0,0,544,537,1,0,0,0,544,538,1,0,0,0,544,539,
  	1,0,0,0,544,540,1,0,0,0,545,85,1,0,0,0,71,89,96,102,106,108,113,128,135,
  	142,148,153,158,163,167,174,180,192,201,205,207,212,224,228,230,233,245,
  	249,251,255,260,272,276,278,282,284,296,300,302,306,320,324,326,330,334,
  	342,347,355,360,370,385,390,393,400,411,422,430,438,446,454,462,470,478,
  	486,494,500,505,511,526,529,532,544
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  zenithparserParserStaticData = std::move(staticData);
}

}

ZenithParser::ZenithParser(TokenStream *input) : ZenithParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

ZenithParser::ZenithParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  ZenithParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *zenithparserParserStaticData->atn, zenithparserParserStaticData->decisionToDFA, zenithparserParserStaticData->sharedContextCache, options);
}

ZenithParser::~ZenithParser() {
  delete _interpreter;
}

const atn::ATN& ZenithParser::getATN() const {
  return *zenithparserParserStaticData->atn;
}

std::string ZenithParser::getGrammarFileName() const {
  return "ZenithParser.g4";
}

const std::vector<std::string>& ZenithParser::getRuleNames() const {
  return zenithparserParserStaticData->ruleNames;
}

const dfa::Vocabulary& ZenithParser::getVocabulary() const {
  return zenithparserParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView ZenithParser::getSerializedATN() const {
  return zenithparserParserStaticData->serializedATN;
}


//----------------- ProgramContext ------------------------------------------------------------------

ZenithParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::ProgramContext::EOF() {
  return getToken(ZenithParser::EOF, 0);
}

std::vector<tree::TerminalNode *> ZenithParser::ProgramContext::NEWLINE() {
  return getTokens(ZenithParser::NEWLINE);
}

tree::TerminalNode* ZenithParser::ProgramContext::NEWLINE(size_t i) {
  return getToken(ZenithParser::NEWLINE, i);
}

std::vector<ZenithParser::StatementContext *> ZenithParser::ProgramContext::statement() {
  return getRuleContexts<ZenithParser::StatementContext>();
}

ZenithParser::StatementContext* ZenithParser::ProgramContext::statement(size_t i) {
  return getRuleContext<ZenithParser::StatementContext>(i);
}

std::vector<ZenithParser::SemiContext *> ZenithParser::ProgramContext::semi() {
  return getRuleContexts<ZenithParser::SemiContext>();
}

ZenithParser::SemiContext* ZenithParser::ProgramContext::semi(size_t i) {
  return getRuleContext<ZenithParser::SemiContext>(i);
}


size_t ZenithParser::ProgramContext::getRuleIndex() const {
  return ZenithParser::RuleProgram;
}


std::any ZenithParser::ProgramContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitProgram(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ProgramContext* ZenithParser::program() {
  ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, getState());
  enterRule(_localctx, 0, ZenithParser::RuleProgram);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(89);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(86);
        match(ZenithParser::NEWLINE); 
      }
      setState(91);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx);
    }
    setState(108);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
    case 1: {
      setState(92);
      statement();
      setState(102);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(94); 
          _errHandler->sync(this);
          _la = _input->LA(1);
          do {
            setState(93);
            semi();
            setState(96); 
            _errHandler->sync(this);
            _la = _input->LA(1);
          } while (_la == ZenithParser::SEMICOLON);
          setState(98);
          statement(); 
        }
        setState(104);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx);
      }
      setState(106);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == ZenithParser::SEMICOLON) {
        setState(105);
        semi();
      }
      break;
    }

    default:
      break;
    }
    setState(113);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::NEWLINE) {
      setState(110);
      match(ZenithParser::NEWLINE);
      setState(115);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(116);
    match(ZenithParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

ZenithParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::VarDeclarationContext* ZenithParser::StatementContext::varDeclaration() {
  return getRuleContext<ZenithParser::VarDeclarationContext>(0);
}

ZenithParser::FunctionDeclContext* ZenithParser::StatementContext::functionDecl() {
  return getRuleContext<ZenithParser::FunctionDeclContext>(0);
}

ZenithParser::EquationContext* ZenithParser::StatementContext::equation() {
  return getRuleContext<ZenithParser::EquationContext>(0);
}

ZenithParser::IfStatementContext* ZenithParser::StatementContext::ifStatement() {
  return getRuleContext<ZenithParser::IfStatementContext>(0);
}

ZenithParser::WhileStatementContext* ZenithParser::StatementContext::whileStatement() {
  return getRuleContext<ZenithParser::WhileStatementContext>(0);
}

ZenithParser::ForStatementContext* ZenithParser::StatementContext::forStatement() {
  return getRuleContext<ZenithParser::ForStatementContext>(0);
}

ZenithParser::ReturnStatementContext* ZenithParser::StatementContext::returnStatement() {
  return getRuleContext<ZenithParser::ReturnStatementContext>(0);
}

ZenithParser::PrintStatementContext* ZenithParser::StatementContext::printStatement() {
  return getRuleContext<ZenithParser::PrintStatementContext>(0);
}

ZenithParser::ExprStatementContext* ZenithParser::StatementContext::exprStatement() {
  return getRuleContext<ZenithParser::ExprStatementContext>(0);
}

ZenithParser::BlockStatementContext* ZenithParser::StatementContext::blockStatement() {
  return getRuleContext<ZenithParser::BlockStatementContext>(0);
}


size_t ZenithParser::StatementContext::getRuleIndex() const {
  return ZenithParser::RuleStatement;
}


std::any ZenithParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::StatementContext* ZenithParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 2, ZenithParser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(128);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(118);
      varDeclaration();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(119);
      functionDecl();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(120);
      equation();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(121);
      ifStatement();
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(122);
      whileStatement();
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(123);
      forStatement();
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(124);
      returnStatement();
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(125);
      printStatement();
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(126);
      exprStatement();
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(127);
      blockStatement();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SemiContext ------------------------------------------------------------------

ZenithParser::SemiContext::SemiContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::SemiContext::SEMICOLON() {
  return getToken(ZenithParser::SEMICOLON, 0);
}


size_t ZenithParser::SemiContext::getRuleIndex() const {
  return ZenithParser::RuleSemi;
}


std::any ZenithParser::SemiContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitSemi(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::SemiContext* ZenithParser::semi() {
  SemiContext *_localctx = _tracker.createInstance<SemiContext>(_ctx, getState());
  enterRule(_localctx, 4, ZenithParser::RuleSemi);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(130);
    match(ZenithParser::SEMICOLON);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarDeclarationContext ------------------------------------------------------------------

ZenithParser::VarDeclarationContext::VarDeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::IdentifierListContext* ZenithParser::VarDeclarationContext::identifierList() {
  return getRuleContext<ZenithParser::IdentifierListContext>(0);
}

tree::TerminalNode* ZenithParser::VarDeclarationContext::COLON() {
  return getToken(ZenithParser::COLON, 0);
}

ZenithParser::TypeContext* ZenithParser::VarDeclarationContext::type() {
  return getRuleContext<ZenithParser::TypeContext>(0);
}


size_t ZenithParser::VarDeclarationContext::getRuleIndex() const {
  return ZenithParser::RuleVarDeclaration;
}


std::any ZenithParser::VarDeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitVarDeclaration(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::VarDeclarationContext* ZenithParser::varDeclaration() {
  VarDeclarationContext *_localctx = _tracker.createInstance<VarDeclarationContext>(_ctx, getState());
  enterRule(_localctx, 6, ZenithParser::RuleVarDeclaration);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(132);
    identifierList();
    setState(135);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ZenithParser::COLON) {
      setState(133);
      match(ZenithParser::COLON);
      setState(134);
      type();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdentifierListContext ------------------------------------------------------------------

ZenithParser::IdentifierListContext::IdentifierListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> ZenithParser::IdentifierListContext::IDENTIFIER() {
  return getTokens(ZenithParser::IDENTIFIER);
}

tree::TerminalNode* ZenithParser::IdentifierListContext::IDENTIFIER(size_t i) {
  return getToken(ZenithParser::IDENTIFIER, i);
}

std::vector<tree::TerminalNode *> ZenithParser::IdentifierListContext::COMMA() {
  return getTokens(ZenithParser::COMMA);
}

tree::TerminalNode* ZenithParser::IdentifierListContext::COMMA(size_t i) {
  return getToken(ZenithParser::COMMA, i);
}


size_t ZenithParser::IdentifierListContext::getRuleIndex() const {
  return ZenithParser::RuleIdentifierList;
}


std::any ZenithParser::IdentifierListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitIdentifierList(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::IdentifierListContext* ZenithParser::identifierList() {
  IdentifierListContext *_localctx = _tracker.createInstance<IdentifierListContext>(_ctx, getState());
  enterRule(_localctx, 8, ZenithParser::RuleIdentifierList);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(137);
    match(ZenithParser::IDENTIFIER);
    setState(142);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::COMMA) {
      setState(138);
      match(ZenithParser::COMMA);
      setState(139);
      match(ZenithParser::IDENTIFIER);
      setState(144);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionDeclContext ------------------------------------------------------------------

ZenithParser::FunctionDeclContext::FunctionDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::FunctionDeclContext::IDENTIFIER() {
  return getToken(ZenithParser::IDENTIFIER, 0);
}

tree::TerminalNode* ZenithParser::FunctionDeclContext::LPAREN() {
  return getToken(ZenithParser::LPAREN, 0);
}

tree::TerminalNode* ZenithParser::FunctionDeclContext::RPAREN() {
  return getToken(ZenithParser::RPAREN, 0);
}

tree::TerminalNode* ZenithParser::FunctionDeclContext::EQUALS() {
  return getToken(ZenithParser::EQUALS, 0);
}

ZenithParser::ExpressionContext* ZenithParser::FunctionDeclContext::expression() {
  return getRuleContext<ZenithParser::ExpressionContext>(0);
}

ZenithParser::BlockStatementContext* ZenithParser::FunctionDeclContext::blockStatement() {
  return getRuleContext<ZenithParser::BlockStatementContext>(0);
}

ZenithParser::ParameterListContext* ZenithParser::FunctionDeclContext::parameterList() {
  return getRuleContext<ZenithParser::ParameterListContext>(0);
}

tree::TerminalNode* ZenithParser::FunctionDeclContext::ARROW() {
  return getToken(ZenithParser::ARROW, 0);
}

ZenithParser::TypeContext* ZenithParser::FunctionDeclContext::type() {
  return getRuleContext<ZenithParser::TypeContext>(0);
}

std::vector<tree::TerminalNode *> ZenithParser::FunctionDeclContext::NEWLINE() {
  return getTokens(ZenithParser::NEWLINE);
}

tree::TerminalNode* ZenithParser::FunctionDeclContext::NEWLINE(size_t i) {
  return getToken(ZenithParser::NEWLINE, i);
}


size_t ZenithParser::FunctionDeclContext::getRuleIndex() const {
  return ZenithParser::RuleFunctionDecl;
}


std::any ZenithParser::FunctionDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitFunctionDecl(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::FunctionDeclContext* ZenithParser::functionDecl() {
  FunctionDeclContext *_localctx = _tracker.createInstance<FunctionDeclContext>(_ctx, getState());
  enterRule(_localctx, 10, ZenithParser::RuleFunctionDecl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(145);
    match(ZenithParser::IDENTIFIER);
    setState(146);
    match(ZenithParser::LPAREN);
    setState(148);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ZenithParser::IDENTIFIER) {
      setState(147);
      parameterList();
    }
    setState(150);
    match(ZenithParser::RPAREN);
    setState(153);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ZenithParser::ARROW) {
      setState(151);
      match(ZenithParser::ARROW);
      setState(152);
      type();
    }
    setState(167);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx)) {
    case 1: {
      setState(155);
      match(ZenithParser::EQUALS);
      setState(156);
      expression();
      break;
    }

    case 2: {
      setState(158);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if (_la == ZenithParser::EQUALS) {
        setState(157);
        match(ZenithParser::EQUALS);
      }
      setState(163);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(160);
          match(ZenithParser::NEWLINE); 
        }
        setState(165);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
      }
      setState(166);
      blockStatement();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ParameterListContext ------------------------------------------------------------------

ZenithParser::ParameterListContext::ParameterListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::ParameterContext *> ZenithParser::ParameterListContext::parameter() {
  return getRuleContexts<ZenithParser::ParameterContext>();
}

ZenithParser::ParameterContext* ZenithParser::ParameterListContext::parameter(size_t i) {
  return getRuleContext<ZenithParser::ParameterContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::ParameterListContext::COMMA() {
  return getTokens(ZenithParser::COMMA);
}

tree::TerminalNode* ZenithParser::ParameterListContext::COMMA(size_t i) {
  return getToken(ZenithParser::COMMA, i);
}


size_t ZenithParser::ParameterListContext::getRuleIndex() const {
  return ZenithParser::RuleParameterList;
}


std::any ZenithParser::ParameterListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitParameterList(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ParameterListContext* ZenithParser::parameterList() {
  ParameterListContext *_localctx = _tracker.createInstance<ParameterListContext>(_ctx, getState());
  enterRule(_localctx, 12, ZenithParser::RuleParameterList);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(169);
    parameter();
    setState(174);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::COMMA) {
      setState(170);
      match(ZenithParser::COMMA);
      setState(171);
      parameter();
      setState(176);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ParameterContext ------------------------------------------------------------------

ZenithParser::ParameterContext::ParameterContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::ParameterContext::IDENTIFIER() {
  return getToken(ZenithParser::IDENTIFIER, 0);
}

tree::TerminalNode* ZenithParser::ParameterContext::COLON() {
  return getToken(ZenithParser::COLON, 0);
}

ZenithParser::TypeContext* ZenithParser::ParameterContext::type() {
  return getRuleContext<ZenithParser::TypeContext>(0);
}


size_t ZenithParser::ParameterContext::getRuleIndex() const {
  return ZenithParser::RuleParameter;
}


std::any ZenithParser::ParameterContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitParameter(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ParameterContext* ZenithParser::parameter() {
  ParameterContext *_localctx = _tracker.createInstance<ParameterContext>(_ctx, getState());
  enterRule(_localctx, 14, ZenithParser::RuleParameter);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(177);
    match(ZenithParser::IDENTIFIER);
    setState(180);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ZenithParser::COLON) {
      setState(178);
      match(ZenithParser::COLON);
      setState(179);
      type();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EquationContext ------------------------------------------------------------------

ZenithParser::EquationContext::EquationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::ExpressionContext *> ZenithParser::EquationContext::expression() {
  return getRuleContexts<ZenithParser::ExpressionContext>();
}

ZenithParser::ExpressionContext* ZenithParser::EquationContext::expression(size_t i) {
  return getRuleContext<ZenithParser::ExpressionContext>(i);
}

tree::TerminalNode* ZenithParser::EquationContext::EQUALS() {
  return getToken(ZenithParser::EQUALS, 0);
}


size_t ZenithParser::EquationContext::getRuleIndex() const {
  return ZenithParser::RuleEquation;
}


std::any ZenithParser::EquationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitEquation(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::EquationContext* ZenithParser::equation() {
  EquationContext *_localctx = _tracker.createInstance<EquationContext>(_ctx, getState());
  enterRule(_localctx, 16, ZenithParser::RuleEquation);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(182);
    expression();
    setState(183);
    match(ZenithParser::EQUALS);
    setState(184);
    expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExprStatementContext ------------------------------------------------------------------

ZenithParser::ExprStatementContext::ExprStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::ExpressionContext* ZenithParser::ExprStatementContext::expression() {
  return getRuleContext<ZenithParser::ExpressionContext>(0);
}


size_t ZenithParser::ExprStatementContext::getRuleIndex() const {
  return ZenithParser::RuleExprStatement;
}


std::any ZenithParser::ExprStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitExprStatement(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ExprStatementContext* ZenithParser::exprStatement() {
  ExprStatementContext *_localctx = _tracker.createInstance<ExprStatementContext>(_ctx, getState());
  enterRule(_localctx, 18, ZenithParser::RuleExprStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(186);
    expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockStatementContext ------------------------------------------------------------------

ZenithParser::BlockStatementContext::BlockStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::BlockStatementContext::LBRACE() {
  return getToken(ZenithParser::LBRACE, 0);
}

tree::TerminalNode* ZenithParser::BlockStatementContext::RBRACE() {
  return getToken(ZenithParser::RBRACE, 0);
}

std::vector<tree::TerminalNode *> ZenithParser::BlockStatementContext::NEWLINE() {
  return getTokens(ZenithParser::NEWLINE);
}

tree::TerminalNode* ZenithParser::BlockStatementContext::NEWLINE(size_t i) {
  return getToken(ZenithParser::NEWLINE, i);
}

std::vector<ZenithParser::StatementContext *> ZenithParser::BlockStatementContext::statement() {
  return getRuleContexts<ZenithParser::StatementContext>();
}

ZenithParser::StatementContext* ZenithParser::BlockStatementContext::statement(size_t i) {
  return getRuleContext<ZenithParser::StatementContext>(i);
}

std::vector<ZenithParser::SemiContext *> ZenithParser::BlockStatementContext::semi() {
  return getRuleContexts<ZenithParser::SemiContext>();
}

ZenithParser::SemiContext* ZenithParser::BlockStatementContext::semi(size_t i) {
  return getRuleContext<ZenithParser::SemiContext>(i);
}

tree::TerminalNode* ZenithParser::BlockStatementContext::INDENT() {
  return getToken(ZenithParser::INDENT, 0);
}

tree::TerminalNode* ZenithParser::BlockStatementContext::DEDENT() {
  return getToken(ZenithParser::DEDENT, 0);
}


size_t ZenithParser::BlockStatementContext::getRuleIndex() const {
  return ZenithParser::RuleBlockStatement;
}


std::any ZenithParser::BlockStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitBlockStatement(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::BlockStatementContext* ZenithParser::blockStatement() {
  BlockStatementContext *_localctx = _tracker.createInstance<BlockStatementContext>(_ctx, getState());
  enterRule(_localctx, 20, ZenithParser::RuleBlockStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    setState(233);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ZenithParser::LBRACE: {
        enterOuterAlt(_localctx, 1);
        setState(188);
        match(ZenithParser::LBRACE);
        setState(192);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(189);
            match(ZenithParser::NEWLINE); 
          }
          setState(194);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx);
        }
        setState(207);
        _errHandler->sync(this);

        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
        case 1: {
          setState(195);
          statement();
          setState(201);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx);
          while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
            if (alt == 1) {
              setState(196);
              semi();
              setState(197);
              statement(); 
            }
            setState(203);
            _errHandler->sync(this);
            alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx);
          }
          setState(205);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == ZenithParser::SEMICOLON) {
            setState(204);
            semi();
          }
          break;
        }

        default:
          break;
        }
        setState(212);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == ZenithParser::NEWLINE) {
          setState(209);
          match(ZenithParser::NEWLINE);
          setState(214);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(215);
        match(ZenithParser::RBRACE);
        break;
      }

      case ZenithParser::NEWLINE: {
        enterOuterAlt(_localctx, 2);
        setState(216);
        match(ZenithParser::NEWLINE);
        setState(217);
        match(ZenithParser::INDENT);
        setState(230);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 17451802354064104) != 0)) {
          setState(218);
          statement();
          setState(224);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx);
          while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
            if (alt == 1) {
              setState(219);
              semi();
              setState(220);
              statement(); 
            }
            setState(226);
            _errHandler->sync(this);
            alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx);
          }
          setState(228);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == ZenithParser::SEMICOLON) {
            setState(227);
            semi();
          }
        }
        setState(232);
        match(ZenithParser::DEDENT);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IfStatementContext ------------------------------------------------------------------

ZenithParser::IfStatementContext::IfStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::IfStatementContext::IF() {
  return getToken(ZenithParser::IF, 0);
}

ZenithParser::ExpressionContext* ZenithParser::IfStatementContext::expression() {
  return getRuleContext<ZenithParser::ExpressionContext>(0);
}

std::vector<tree::TerminalNode *> ZenithParser::IfStatementContext::NEWLINE() {
  return getTokens(ZenithParser::NEWLINE);
}

tree::TerminalNode* ZenithParser::IfStatementContext::NEWLINE(size_t i) {
  return getToken(ZenithParser::NEWLINE, i);
}

std::vector<tree::TerminalNode *> ZenithParser::IfStatementContext::INDENT() {
  return getTokens(ZenithParser::INDENT);
}

tree::TerminalNode* ZenithParser::IfStatementContext::INDENT(size_t i) {
  return getToken(ZenithParser::INDENT, i);
}

std::vector<tree::TerminalNode *> ZenithParser::IfStatementContext::DEDENT() {
  return getTokens(ZenithParser::DEDENT);
}

tree::TerminalNode* ZenithParser::IfStatementContext::DEDENT(size_t i) {
  return getToken(ZenithParser::DEDENT, i);
}

std::vector<ZenithParser::BlockStatementContext *> ZenithParser::IfStatementContext::blockStatement() {
  return getRuleContexts<ZenithParser::BlockStatementContext>();
}

ZenithParser::BlockStatementContext* ZenithParser::IfStatementContext::blockStatement(size_t i) {
  return getRuleContext<ZenithParser::BlockStatementContext>(i);
}

tree::TerminalNode* ZenithParser::IfStatementContext::ELSE() {
  return getToken(ZenithParser::ELSE, 0);
}

std::vector<ZenithParser::StatementContext *> ZenithParser::IfStatementContext::statement() {
  return getRuleContexts<ZenithParser::StatementContext>();
}

ZenithParser::StatementContext* ZenithParser::IfStatementContext::statement(size_t i) {
  return getRuleContext<ZenithParser::StatementContext>(i);
}

std::vector<ZenithParser::SemiContext *> ZenithParser::IfStatementContext::semi() {
  return getRuleContexts<ZenithParser::SemiContext>();
}

ZenithParser::SemiContext* ZenithParser::IfStatementContext::semi(size_t i) {
  return getRuleContext<ZenithParser::SemiContext>(i);
}


size_t ZenithParser::IfStatementContext::getRuleIndex() const {
  return ZenithParser::RuleIfStatement;
}


std::any ZenithParser::IfStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitIfStatement(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::IfStatementContext* ZenithParser::ifStatement() {
  IfStatementContext *_localctx = _tracker.createInstance<IfStatementContext>(_ctx, getState());
  enterRule(_localctx, 22, ZenithParser::RuleIfStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(235);
    match(ZenithParser::IF);
    setState(236);
    expression();
    setState(255);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 28, _ctx)) {
    case 1: {
      setState(237);
      match(ZenithParser::NEWLINE);
      setState(238);
      match(ZenithParser::INDENT);
      setState(251);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 17451802354064104) != 0)) {
        setState(239);
        statement();
        setState(245);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(240);
            semi();
            setState(241);
            statement(); 
          }
          setState(247);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx);
        }
        setState(249);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == ZenithParser::SEMICOLON) {
          setState(248);
          semi();
        }
      }
      setState(253);
      match(ZenithParser::DEDENT);
      break;
    }

    case 2: {
      setState(254);
      blockStatement();
      break;
    }

    default:
      break;
    }
    setState(284);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx)) {
    case 1: {
      setState(260);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == ZenithParser::NEWLINE) {
        setState(257);
        match(ZenithParser::NEWLINE);
        setState(262);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(263);
      match(ZenithParser::ELSE);
      setState(282);
      _errHandler->sync(this);
      switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx)) {
      case 1: {
        setState(264);
        match(ZenithParser::NEWLINE);
        setState(265);
        match(ZenithParser::INDENT);
        setState(278);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 17451802354064104) != 0)) {
          setState(266);
          statement();
          setState(272);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx);
          while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
            if (alt == 1) {
              setState(267);
              semi();
              setState(268);
              statement(); 
            }
            setState(274);
            _errHandler->sync(this);
            alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx);
          }
          setState(276);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if (_la == ZenithParser::SEMICOLON) {
            setState(275);
            semi();
          }
        }
        setState(280);
        match(ZenithParser::DEDENT);
        break;
      }

      case 2: {
        setState(281);
        blockStatement();
        break;
      }

      default:
        break;
      }
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- WhileStatementContext ------------------------------------------------------------------

ZenithParser::WhileStatementContext::WhileStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::WhileStatementContext::WHILE() {
  return getToken(ZenithParser::WHILE, 0);
}

ZenithParser::ExpressionContext* ZenithParser::WhileStatementContext::expression() {
  return getRuleContext<ZenithParser::ExpressionContext>(0);
}

tree::TerminalNode* ZenithParser::WhileStatementContext::NEWLINE() {
  return getToken(ZenithParser::NEWLINE, 0);
}

tree::TerminalNode* ZenithParser::WhileStatementContext::INDENT() {
  return getToken(ZenithParser::INDENT, 0);
}

tree::TerminalNode* ZenithParser::WhileStatementContext::DEDENT() {
  return getToken(ZenithParser::DEDENT, 0);
}

ZenithParser::BlockStatementContext* ZenithParser::WhileStatementContext::blockStatement() {
  return getRuleContext<ZenithParser::BlockStatementContext>(0);
}

std::vector<ZenithParser::StatementContext *> ZenithParser::WhileStatementContext::statement() {
  return getRuleContexts<ZenithParser::StatementContext>();
}

ZenithParser::StatementContext* ZenithParser::WhileStatementContext::statement(size_t i) {
  return getRuleContext<ZenithParser::StatementContext>(i);
}

std::vector<ZenithParser::SemiContext *> ZenithParser::WhileStatementContext::semi() {
  return getRuleContexts<ZenithParser::SemiContext>();
}

ZenithParser::SemiContext* ZenithParser::WhileStatementContext::semi(size_t i) {
  return getRuleContext<ZenithParser::SemiContext>(i);
}


size_t ZenithParser::WhileStatementContext::getRuleIndex() const {
  return ZenithParser::RuleWhileStatement;
}


std::any ZenithParser::WhileStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitWhileStatement(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::WhileStatementContext* ZenithParser::whileStatement() {
  WhileStatementContext *_localctx = _tracker.createInstance<WhileStatementContext>(_ctx, getState());
  enterRule(_localctx, 24, ZenithParser::RuleWhileStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(286);
    match(ZenithParser::WHILE);
    setState(287);
    expression();
    setState(306);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 38, _ctx)) {
    case 1: {
      setState(288);
      match(ZenithParser::NEWLINE);
      setState(289);
      match(ZenithParser::INDENT);
      setState(302);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 17451802354064104) != 0)) {
        setState(290);
        statement();
        setState(296);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 35, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(291);
            semi();
            setState(292);
            statement(); 
          }
          setState(298);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 35, _ctx);
        }
        setState(300);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == ZenithParser::SEMICOLON) {
          setState(299);
          semi();
        }
      }
      setState(304);
      match(ZenithParser::DEDENT);
      break;
    }

    case 2: {
      setState(305);
      blockStatement();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ForStatementContext ------------------------------------------------------------------

ZenithParser::ForStatementContext::ForStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::ForStatementContext::FOR() {
  return getToken(ZenithParser::FOR, 0);
}

tree::TerminalNode* ZenithParser::ForStatementContext::IDENTIFIER() {
  return getToken(ZenithParser::IDENTIFIER, 0);
}

tree::TerminalNode* ZenithParser::ForStatementContext::IN() {
  return getToken(ZenithParser::IN, 0);
}

ZenithParser::ExpressionContext* ZenithParser::ForStatementContext::expression() {
  return getRuleContext<ZenithParser::ExpressionContext>(0);
}

tree::TerminalNode* ZenithParser::ForStatementContext::NEWLINE() {
  return getToken(ZenithParser::NEWLINE, 0);
}

tree::TerminalNode* ZenithParser::ForStatementContext::INDENT() {
  return getToken(ZenithParser::INDENT, 0);
}

tree::TerminalNode* ZenithParser::ForStatementContext::DEDENT() {
  return getToken(ZenithParser::DEDENT, 0);
}

ZenithParser::BlockStatementContext* ZenithParser::ForStatementContext::blockStatement() {
  return getRuleContext<ZenithParser::BlockStatementContext>(0);
}

std::vector<ZenithParser::StatementContext *> ZenithParser::ForStatementContext::statement() {
  return getRuleContexts<ZenithParser::StatementContext>();
}

ZenithParser::StatementContext* ZenithParser::ForStatementContext::statement(size_t i) {
  return getRuleContext<ZenithParser::StatementContext>(i);
}

std::vector<ZenithParser::SemiContext *> ZenithParser::ForStatementContext::semi() {
  return getRuleContexts<ZenithParser::SemiContext>();
}

ZenithParser::SemiContext* ZenithParser::ForStatementContext::semi(size_t i) {
  return getRuleContext<ZenithParser::SemiContext>(i);
}


size_t ZenithParser::ForStatementContext::getRuleIndex() const {
  return ZenithParser::RuleForStatement;
}


std::any ZenithParser::ForStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitForStatement(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ForStatementContext* ZenithParser::forStatement() {
  ForStatementContext *_localctx = _tracker.createInstance<ForStatementContext>(_ctx, getState());
  enterRule(_localctx, 26, ZenithParser::RuleForStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(308);
    match(ZenithParser::FOR);
    setState(309);
    match(ZenithParser::IDENTIFIER);
    setState(310);
    match(ZenithParser::IN);
    setState(311);
    expression();
    setState(330);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 42, _ctx)) {
    case 1: {
      setState(312);
      match(ZenithParser::NEWLINE);
      setState(313);
      match(ZenithParser::INDENT);
      setState(326);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 17451802354064104) != 0)) {
        setState(314);
        statement();
        setState(320);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 39, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            setState(315);
            semi();
            setState(316);
            statement(); 
          }
          setState(322);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 39, _ctx);
        }
        setState(324);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == ZenithParser::SEMICOLON) {
          setState(323);
          semi();
        }
      }
      setState(328);
      match(ZenithParser::DEDENT);
      break;
    }

    case 2: {
      setState(329);
      blockStatement();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReturnStatementContext ------------------------------------------------------------------

ZenithParser::ReturnStatementContext::ReturnStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::ReturnStatementContext::RETURN() {
  return getToken(ZenithParser::RETURN, 0);
}

ZenithParser::ExpressionContext* ZenithParser::ReturnStatementContext::expression() {
  return getRuleContext<ZenithParser::ExpressionContext>(0);
}


size_t ZenithParser::ReturnStatementContext::getRuleIndex() const {
  return ZenithParser::RuleReturnStatement;
}


std::any ZenithParser::ReturnStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitReturnStatement(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ReturnStatementContext* ZenithParser::returnStatement() {
  ReturnStatementContext *_localctx = _tracker.createInstance<ReturnStatementContext>(_ctx, getState());
  enterRule(_localctx, 28, ZenithParser::RuleReturnStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(332);
    match(ZenithParser::RETURN);
    setState(334);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 8444328221415424) != 0)) {
      setState(333);
      expression();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrintStatementContext ------------------------------------------------------------------

ZenithParser::PrintStatementContext::PrintStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::PrintStatementContext::PRINT() {
  return getToken(ZenithParser::PRINT, 0);
}

std::vector<ZenithParser::ExpressionContext *> ZenithParser::PrintStatementContext::expression() {
  return getRuleContexts<ZenithParser::ExpressionContext>();
}

ZenithParser::ExpressionContext* ZenithParser::PrintStatementContext::expression(size_t i) {
  return getRuleContext<ZenithParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::PrintStatementContext::COMMA() {
  return getTokens(ZenithParser::COMMA);
}

tree::TerminalNode* ZenithParser::PrintStatementContext::COMMA(size_t i) {
  return getToken(ZenithParser::COMMA, i);
}


size_t ZenithParser::PrintStatementContext::getRuleIndex() const {
  return ZenithParser::RulePrintStatement;
}


std::any ZenithParser::PrintStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitPrintStatement(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::PrintStatementContext* ZenithParser::printStatement() {
  PrintStatementContext *_localctx = _tracker.createInstance<PrintStatementContext>(_ctx, getState());
  enterRule(_localctx, 30, ZenithParser::RulePrintStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(336);
    match(ZenithParser::PRINT);
    setState(337);
    expression();
    setState(342);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::COMMA) {
      setState(338);
      match(ZenithParser::COMMA);
      setState(339);
      expression();
      setState(344);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TypeContext ------------------------------------------------------------------

ZenithParser::TypeContext::TypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::BaseTypeContext* ZenithParser::TypeContext::baseType() {
  return getRuleContext<ZenithParser::BaseTypeContext>(0);
}

ZenithParser::DependentPredicateContext* ZenithParser::TypeContext::dependentPredicate() {
  return getRuleContext<ZenithParser::DependentPredicateContext>(0);
}


size_t ZenithParser::TypeContext::getRuleIndex() const {
  return ZenithParser::RuleType;
}


std::any ZenithParser::TypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitType(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::TypeContext* ZenithParser::type() {
  TypeContext *_localctx = _tracker.createInstance<TypeContext>(_ctx, getState());
  enterRule(_localctx, 32, ZenithParser::RuleType);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(345);
    baseType();
    setState(347);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 45, _ctx)) {
    case 1: {
      setState(346);
      dependentPredicate();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BaseTypeContext ------------------------------------------------------------------

ZenithParser::BaseTypeContext::BaseTypeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::BaseTypeContext::IDENTIFIER() {
  return getToken(ZenithParser::IDENTIFIER, 0);
}

std::vector<tree::TerminalNode *> ZenithParser::BaseTypeContext::LBRACKET() {
  return getTokens(ZenithParser::LBRACKET);
}

tree::TerminalNode* ZenithParser::BaseTypeContext::LBRACKET(size_t i) {
  return getToken(ZenithParser::LBRACKET, i);
}

std::vector<tree::TerminalNode *> ZenithParser::BaseTypeContext::INTEGER() {
  return getTokens(ZenithParser::INTEGER);
}

tree::TerminalNode* ZenithParser::BaseTypeContext::INTEGER(size_t i) {
  return getToken(ZenithParser::INTEGER, i);
}

std::vector<tree::TerminalNode *> ZenithParser::BaseTypeContext::RBRACKET() {
  return getTokens(ZenithParser::RBRACKET);
}

tree::TerminalNode* ZenithParser::BaseTypeContext::RBRACKET(size_t i) {
  return getToken(ZenithParser::RBRACKET, i);
}

tree::TerminalNode* ZenithParser::BaseTypeContext::AMPERSAND() {
  return getToken(ZenithParser::AMPERSAND, 0);
}

ZenithParser::BaseTypeContext* ZenithParser::BaseTypeContext::baseType() {
  return getRuleContext<ZenithParser::BaseTypeContext>(0);
}


size_t ZenithParser::BaseTypeContext::getRuleIndex() const {
  return ZenithParser::RuleBaseType;
}


std::any ZenithParser::BaseTypeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitBaseType(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::BaseTypeContext* ZenithParser::baseType() {
  BaseTypeContext *_localctx = _tracker.createInstance<BaseTypeContext>(_ctx, getState());
  enterRule(_localctx, 34, ZenithParser::RuleBaseType);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(360);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ZenithParser::IDENTIFIER: {
        enterOuterAlt(_localctx, 1);
        setState(349);
        match(ZenithParser::IDENTIFIER);
        setState(355);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == ZenithParser::LBRACKET) {
          setState(350);
          match(ZenithParser::LBRACKET);
          setState(351);
          match(ZenithParser::INTEGER);
          setState(352);
          match(ZenithParser::RBRACKET);
          setState(357);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        break;
      }

      case ZenithParser::AMPERSAND: {
        enterOuterAlt(_localctx, 2);
        setState(358);
        match(ZenithParser::AMPERSAND);
        setState(359);
        baseType();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DependentPredicateContext ------------------------------------------------------------------

ZenithParser::DependentPredicateContext::DependentPredicateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::DependentPredicateContext::LBRACE() {
  return getToken(ZenithParser::LBRACE, 0);
}

ZenithParser::PredicateContext* ZenithParser::DependentPredicateContext::predicate() {
  return getRuleContext<ZenithParser::PredicateContext>(0);
}

tree::TerminalNode* ZenithParser::DependentPredicateContext::RBRACE() {
  return getToken(ZenithParser::RBRACE, 0);
}


size_t ZenithParser::DependentPredicateContext::getRuleIndex() const {
  return ZenithParser::RuleDependentPredicate;
}


std::any ZenithParser::DependentPredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitDependentPredicate(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::DependentPredicateContext* ZenithParser::dependentPredicate() {
  DependentPredicateContext *_localctx = _tracker.createInstance<DependentPredicateContext>(_ctx, getState());
  enterRule(_localctx, 36, ZenithParser::RuleDependentPredicate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(362);
    match(ZenithParser::LBRACE);
    setState(363);
    predicate();
    setState(364);
    match(ZenithParser::RBRACE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PredicateContext ------------------------------------------------------------------

ZenithParser::PredicateContext::PredicateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::RangePredicateContext* ZenithParser::PredicateContext::rangePredicate() {
  return getRuleContext<ZenithParser::RangePredicateContext>(0);
}

ZenithParser::UnaryPredicateContext* ZenithParser::PredicateContext::unaryPredicate() {
  return getRuleContext<ZenithParser::UnaryPredicateContext>(0);
}

ZenithParser::BinaryPredicateContext* ZenithParser::PredicateContext::binaryPredicate() {
  return getRuleContext<ZenithParser::BinaryPredicateContext>(0);
}

ZenithParser::ComplexPredicateContext* ZenithParser::PredicateContext::complexPredicate() {
  return getRuleContext<ZenithParser::ComplexPredicateContext>(0);
}


size_t ZenithParser::PredicateContext::getRuleIndex() const {
  return ZenithParser::RulePredicate;
}


std::any ZenithParser::PredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitPredicate(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::PredicateContext* ZenithParser::predicate() {
  PredicateContext *_localctx = _tracker.createInstance<PredicateContext>(_ctx, getState());
  enterRule(_localctx, 38, ZenithParser::RulePredicate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(370);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 48, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(366);
      rangePredicate();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(367);
      unaryPredicate();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(368);
      binaryPredicate();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(369);
      complexPredicate();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RangePredicateContext ------------------------------------------------------------------

ZenithParser::RangePredicateContext::RangePredicateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::PredicateValueContext *> ZenithParser::RangePredicateContext::predicateValue() {
  return getRuleContexts<ZenithParser::PredicateValueContext>();
}

ZenithParser::PredicateValueContext* ZenithParser::RangePredicateContext::predicateValue(size_t i) {
  return getRuleContext<ZenithParser::PredicateValueContext>(i);
}

tree::TerminalNode* ZenithParser::RangePredicateContext::DOTDOT() {
  return getToken(ZenithParser::DOTDOT, 0);
}


size_t ZenithParser::RangePredicateContext::getRuleIndex() const {
  return ZenithParser::RuleRangePredicate;
}


std::any ZenithParser::RangePredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitRangePredicate(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::RangePredicateContext* ZenithParser::rangePredicate() {
  RangePredicateContext *_localctx = _tracker.createInstance<RangePredicateContext>(_ctx, getState());
  enterRule(_localctx, 40, ZenithParser::RuleRangePredicate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(372);
    predicateValue();
    setState(373);
    match(ZenithParser::DOTDOT);
    setState(374);
    predicateValue();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryPredicateContext ------------------------------------------------------------------

ZenithParser::UnaryPredicateContext::UnaryPredicateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::PredicateValueContext* ZenithParser::UnaryPredicateContext::predicateValue() {
  return getRuleContext<ZenithParser::PredicateValueContext>(0);
}

tree::TerminalNode* ZenithParser::UnaryPredicateContext::NOT() {
  return getToken(ZenithParser::NOT, 0);
}

tree::TerminalNode* ZenithParser::UnaryPredicateContext::NOT_WORD() {
  return getToken(ZenithParser::NOT_WORD, 0);
}

tree::TerminalNode* ZenithParser::UnaryPredicateContext::MINUS() {
  return getToken(ZenithParser::MINUS, 0);
}


size_t ZenithParser::UnaryPredicateContext::getRuleIndex() const {
  return ZenithParser::RuleUnaryPredicate;
}


std::any ZenithParser::UnaryPredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitUnaryPredicate(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::UnaryPredicateContext* ZenithParser::unaryPredicate() {
  UnaryPredicateContext *_localctx = _tracker.createInstance<UnaryPredicateContext>(_ctx, getState());
  enterRule(_localctx, 42, ZenithParser::RuleUnaryPredicate);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(376);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 536952832) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(377);
    predicateValue();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BinaryPredicateContext ------------------------------------------------------------------

ZenithParser::BinaryPredicateContext::BinaryPredicateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::PredicateValueContext *> ZenithParser::BinaryPredicateContext::predicateValue() {
  return getRuleContexts<ZenithParser::PredicateValueContext>();
}

ZenithParser::PredicateValueContext* ZenithParser::BinaryPredicateContext::predicateValue(size_t i) {
  return getRuleContext<ZenithParser::PredicateValueContext>(i);
}

std::vector<ZenithParser::PredicateOpContext *> ZenithParser::BinaryPredicateContext::predicateOp() {
  return getRuleContexts<ZenithParser::PredicateOpContext>();
}

ZenithParser::PredicateOpContext* ZenithParser::BinaryPredicateContext::predicateOp(size_t i) {
  return getRuleContext<ZenithParser::PredicateOpContext>(i);
}


size_t ZenithParser::BinaryPredicateContext::getRuleIndex() const {
  return ZenithParser::RuleBinaryPredicate;
}


std::any ZenithParser::BinaryPredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitBinaryPredicate(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::BinaryPredicateContext* ZenithParser::binaryPredicate() {
  BinaryPredicateContext *_localctx = _tracker.createInstance<BinaryPredicateContext>(_ctx, getState());
  enterRule(_localctx, 44, ZenithParser::RuleBinaryPredicate);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(379);
    predicateValue();
    setState(383); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(380);
      predicateOp();
      setState(381);
      predicateValue();
      setState(385); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 535789568) != 0));
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ComplexPredicateContext ------------------------------------------------------------------

ZenithParser::ComplexPredicateContext::ComplexPredicateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::ComplexPredicateContext::IDENTIFIER() {
  return getToken(ZenithParser::IDENTIFIER, 0);
}

tree::TerminalNode* ZenithParser::ComplexPredicateContext::LPAREN() {
  return getToken(ZenithParser::LPAREN, 0);
}

tree::TerminalNode* ZenithParser::ComplexPredicateContext::RPAREN() {
  return getToken(ZenithParser::RPAREN, 0);
}

ZenithParser::PredicateArgListContext* ZenithParser::ComplexPredicateContext::predicateArgList() {
  return getRuleContext<ZenithParser::PredicateArgListContext>(0);
}


size_t ZenithParser::ComplexPredicateContext::getRuleIndex() const {
  return ZenithParser::RuleComplexPredicate;
}


std::any ZenithParser::ComplexPredicateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitComplexPredicate(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ComplexPredicateContext* ZenithParser::complexPredicate() {
  ComplexPredicateContext *_localctx = _tracker.createInstance<ComplexPredicateContext>(_ctx, getState());
  enterRule(_localctx, 46, ZenithParser::RuleComplexPredicate);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(387);
    match(ZenithParser::IDENTIFIER);
    setState(393);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ZenithParser::LPAREN) {
      setState(388);
      match(ZenithParser::LPAREN);
      setState(390);
      _errHandler->sync(this);

      _la = _input->LA(1);
      if ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 8444318557749248) != 0)) {
        setState(389);
        predicateArgList();
      }
      setState(392);
      match(ZenithParser::RPAREN);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PredicateArgListContext ------------------------------------------------------------------

ZenithParser::PredicateArgListContext::PredicateArgListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::PredicateContext *> ZenithParser::PredicateArgListContext::predicate() {
  return getRuleContexts<ZenithParser::PredicateContext>();
}

ZenithParser::PredicateContext* ZenithParser::PredicateArgListContext::predicate(size_t i) {
  return getRuleContext<ZenithParser::PredicateContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::PredicateArgListContext::COMMA() {
  return getTokens(ZenithParser::COMMA);
}

tree::TerminalNode* ZenithParser::PredicateArgListContext::COMMA(size_t i) {
  return getToken(ZenithParser::COMMA, i);
}


size_t ZenithParser::PredicateArgListContext::getRuleIndex() const {
  return ZenithParser::RulePredicateArgList;
}


std::any ZenithParser::PredicateArgListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitPredicateArgList(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::PredicateArgListContext* ZenithParser::predicateArgList() {
  PredicateArgListContext *_localctx = _tracker.createInstance<PredicateArgListContext>(_ctx, getState());
  enterRule(_localctx, 48, ZenithParser::RulePredicateArgList);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(395);
    predicate();
    setState(400);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::COMMA) {
      setState(396);
      match(ZenithParser::COMMA);
      setState(397);
      predicate();
      setState(402);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PredicateValueContext ------------------------------------------------------------------

ZenithParser::PredicateValueContext::PredicateValueContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::PredicateValueContext::IDENTIFIER() {
  return getToken(ZenithParser::IDENTIFIER, 0);
}

tree::TerminalNode* ZenithParser::PredicateValueContext::INTEGER() {
  return getToken(ZenithParser::INTEGER, 0);
}

tree::TerminalNode* ZenithParser::PredicateValueContext::FLOAT() {
  return getToken(ZenithParser::FLOAT, 0);
}

tree::TerminalNode* ZenithParser::PredicateValueContext::STRING() {
  return getToken(ZenithParser::STRING, 0);
}

tree::TerminalNode* ZenithParser::PredicateValueContext::LPAREN() {
  return getToken(ZenithParser::LPAREN, 0);
}

ZenithParser::PredicateContext* ZenithParser::PredicateValueContext::predicate() {
  return getRuleContext<ZenithParser::PredicateContext>(0);
}

tree::TerminalNode* ZenithParser::PredicateValueContext::RPAREN() {
  return getToken(ZenithParser::RPAREN, 0);
}


size_t ZenithParser::PredicateValueContext::getRuleIndex() const {
  return ZenithParser::RulePredicateValue;
}


std::any ZenithParser::PredicateValueContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitPredicateValue(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::PredicateValueContext* ZenithParser::predicateValue() {
  PredicateValueContext *_localctx = _tracker.createInstance<PredicateValueContext>(_ctx, getState());
  enterRule(_localctx, 50, ZenithParser::RulePredicateValue);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(411);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ZenithParser::IDENTIFIER: {
        enterOuterAlt(_localctx, 1);
        setState(403);
        match(ZenithParser::IDENTIFIER);
        break;
      }

      case ZenithParser::INTEGER: {
        enterOuterAlt(_localctx, 2);
        setState(404);
        match(ZenithParser::INTEGER);
        break;
      }

      case ZenithParser::FLOAT: {
        enterOuterAlt(_localctx, 3);
        setState(405);
        match(ZenithParser::FLOAT);
        break;
      }

      case ZenithParser::STRING: {
        enterOuterAlt(_localctx, 4);
        setState(406);
        match(ZenithParser::STRING);
        break;
      }

      case ZenithParser::LPAREN: {
        enterOuterAlt(_localctx, 5);
        setState(407);
        match(ZenithParser::LPAREN);
        setState(408);
        predicate();
        setState(409);
        match(ZenithParser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PredicateOpContext ------------------------------------------------------------------

ZenithParser::PredicateOpContext::PredicateOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::PredicateOpContext::EQ() {
  return getToken(ZenithParser::EQ, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::NEQ() {
  return getToken(ZenithParser::NEQ, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::LT() {
  return getToken(ZenithParser::LT, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::LE() {
  return getToken(ZenithParser::LE, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::GT() {
  return getToken(ZenithParser::GT, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::GE() {
  return getToken(ZenithParser::GE, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::PLUS() {
  return getToken(ZenithParser::PLUS, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::MINUS() {
  return getToken(ZenithParser::MINUS, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::STAR() {
  return getToken(ZenithParser::STAR, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::DIV() {
  return getToken(ZenithParser::DIV, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::MOD() {
  return getToken(ZenithParser::MOD, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::AND() {
  return getToken(ZenithParser::AND, 0);
}

tree::TerminalNode* ZenithParser::PredicateOpContext::OR() {
  return getToken(ZenithParser::OR, 0);
}


size_t ZenithParser::PredicateOpContext::getRuleIndex() const {
  return ZenithParser::RulePredicateOp;
}


std::any ZenithParser::PredicateOpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitPredicateOp(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::PredicateOpContext* ZenithParser::predicateOp() {
  PredicateOpContext *_localctx = _tracker.createInstance<PredicateOpContext>(_ctx, getState());
  enterRule(_localctx, 52, ZenithParser::RulePredicateOp);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(413);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 535789568) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

ZenithParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::LogicalOrExprContext* ZenithParser::ExpressionContext::logicalOrExpr() {
  return getRuleContext<ZenithParser::LogicalOrExprContext>(0);
}


size_t ZenithParser::ExpressionContext::getRuleIndex() const {
  return ZenithParser::RuleExpression;
}


std::any ZenithParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ExpressionContext* ZenithParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 54, ZenithParser::RuleExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(415);
    logicalOrExpr();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LogicalOrExprContext ------------------------------------------------------------------

ZenithParser::LogicalOrExprContext::LogicalOrExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::LogicalAndExprContext *> ZenithParser::LogicalOrExprContext::logicalAndExpr() {
  return getRuleContexts<ZenithParser::LogicalAndExprContext>();
}

ZenithParser::LogicalAndExprContext* ZenithParser::LogicalOrExprContext::logicalAndExpr(size_t i) {
  return getRuleContext<ZenithParser::LogicalAndExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::LogicalOrExprContext::OR() {
  return getTokens(ZenithParser::OR);
}

tree::TerminalNode* ZenithParser::LogicalOrExprContext::OR(size_t i) {
  return getToken(ZenithParser::OR, i);
}


size_t ZenithParser::LogicalOrExprContext::getRuleIndex() const {
  return ZenithParser::RuleLogicalOrExpr;
}


std::any ZenithParser::LogicalOrExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitLogicalOrExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::LogicalOrExprContext* ZenithParser::logicalOrExpr() {
  LogicalOrExprContext *_localctx = _tracker.createInstance<LogicalOrExprContext>(_ctx, getState());
  enterRule(_localctx, 56, ZenithParser::RuleLogicalOrExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(417);
    logicalAndExpr();
    setState(422);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::OR) {
      setState(418);
      match(ZenithParser::OR);
      setState(419);
      logicalAndExpr();
      setState(424);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LogicalAndExprContext ------------------------------------------------------------------

ZenithParser::LogicalAndExprContext::LogicalAndExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::BitwiseOrExprContext *> ZenithParser::LogicalAndExprContext::bitwiseOrExpr() {
  return getRuleContexts<ZenithParser::BitwiseOrExprContext>();
}

ZenithParser::BitwiseOrExprContext* ZenithParser::LogicalAndExprContext::bitwiseOrExpr(size_t i) {
  return getRuleContext<ZenithParser::BitwiseOrExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::LogicalAndExprContext::AND() {
  return getTokens(ZenithParser::AND);
}

tree::TerminalNode* ZenithParser::LogicalAndExprContext::AND(size_t i) {
  return getToken(ZenithParser::AND, i);
}


size_t ZenithParser::LogicalAndExprContext::getRuleIndex() const {
  return ZenithParser::RuleLogicalAndExpr;
}


std::any ZenithParser::LogicalAndExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitLogicalAndExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::LogicalAndExprContext* ZenithParser::logicalAndExpr() {
  LogicalAndExprContext *_localctx = _tracker.createInstance<LogicalAndExprContext>(_ctx, getState());
  enterRule(_localctx, 58, ZenithParser::RuleLogicalAndExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(425);
    bitwiseOrExpr();
    setState(430);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::AND) {
      setState(426);
      match(ZenithParser::AND);
      setState(427);
      bitwiseOrExpr();
      setState(432);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BitwiseOrExprContext ------------------------------------------------------------------

ZenithParser::BitwiseOrExprContext::BitwiseOrExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::BitwiseXorExprContext *> ZenithParser::BitwiseOrExprContext::bitwiseXorExpr() {
  return getRuleContexts<ZenithParser::BitwiseXorExprContext>();
}

ZenithParser::BitwiseXorExprContext* ZenithParser::BitwiseOrExprContext::bitwiseXorExpr(size_t i) {
  return getRuleContext<ZenithParser::BitwiseXorExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::BitwiseOrExprContext::PIPE() {
  return getTokens(ZenithParser::PIPE);
}

tree::TerminalNode* ZenithParser::BitwiseOrExprContext::PIPE(size_t i) {
  return getToken(ZenithParser::PIPE, i);
}


size_t ZenithParser::BitwiseOrExprContext::getRuleIndex() const {
  return ZenithParser::RuleBitwiseOrExpr;
}


std::any ZenithParser::BitwiseOrExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitBitwiseOrExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::BitwiseOrExprContext* ZenithParser::bitwiseOrExpr() {
  BitwiseOrExprContext *_localctx = _tracker.createInstance<BitwiseOrExprContext>(_ctx, getState());
  enterRule(_localctx, 60, ZenithParser::RuleBitwiseOrExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(433);
    bitwiseXorExpr();
    setState(438);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::PIPE) {
      setState(434);
      match(ZenithParser::PIPE);
      setState(435);
      bitwiseXorExpr();
      setState(440);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BitwiseXorExprContext ------------------------------------------------------------------

ZenithParser::BitwiseXorExprContext::BitwiseXorExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::BitwiseAndExprContext *> ZenithParser::BitwiseXorExprContext::bitwiseAndExpr() {
  return getRuleContexts<ZenithParser::BitwiseAndExprContext>();
}

ZenithParser::BitwiseAndExprContext* ZenithParser::BitwiseXorExprContext::bitwiseAndExpr(size_t i) {
  return getRuleContext<ZenithParser::BitwiseAndExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::BitwiseXorExprContext::CARET() {
  return getTokens(ZenithParser::CARET);
}

tree::TerminalNode* ZenithParser::BitwiseXorExprContext::CARET(size_t i) {
  return getToken(ZenithParser::CARET, i);
}


size_t ZenithParser::BitwiseXorExprContext::getRuleIndex() const {
  return ZenithParser::RuleBitwiseXorExpr;
}


std::any ZenithParser::BitwiseXorExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitBitwiseXorExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::BitwiseXorExprContext* ZenithParser::bitwiseXorExpr() {
  BitwiseXorExprContext *_localctx = _tracker.createInstance<BitwiseXorExprContext>(_ctx, getState());
  enterRule(_localctx, 62, ZenithParser::RuleBitwiseXorExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(441);
    bitwiseAndExpr();
    setState(446);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::CARET) {
      setState(442);
      match(ZenithParser::CARET);
      setState(443);
      bitwiseAndExpr();
      setState(448);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BitwiseAndExprContext ------------------------------------------------------------------

ZenithParser::BitwiseAndExprContext::BitwiseAndExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::EqualityExprContext *> ZenithParser::BitwiseAndExprContext::equalityExpr() {
  return getRuleContexts<ZenithParser::EqualityExprContext>();
}

ZenithParser::EqualityExprContext* ZenithParser::BitwiseAndExprContext::equalityExpr(size_t i) {
  return getRuleContext<ZenithParser::EqualityExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::BitwiseAndExprContext::AMPERSAND() {
  return getTokens(ZenithParser::AMPERSAND);
}

tree::TerminalNode* ZenithParser::BitwiseAndExprContext::AMPERSAND(size_t i) {
  return getToken(ZenithParser::AMPERSAND, i);
}


size_t ZenithParser::BitwiseAndExprContext::getRuleIndex() const {
  return ZenithParser::RuleBitwiseAndExpr;
}


std::any ZenithParser::BitwiseAndExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitBitwiseAndExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::BitwiseAndExprContext* ZenithParser::bitwiseAndExpr() {
  BitwiseAndExprContext *_localctx = _tracker.createInstance<BitwiseAndExprContext>(_ctx, getState());
  enterRule(_localctx, 64, ZenithParser::RuleBitwiseAndExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(449);
    equalityExpr();
    setState(454);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::AMPERSAND) {
      setState(450);
      match(ZenithParser::AMPERSAND);
      setState(451);
      equalityExpr();
      setState(456);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EqualityExprContext ------------------------------------------------------------------

ZenithParser::EqualityExprContext::EqualityExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::RelationalExprContext *> ZenithParser::EqualityExprContext::relationalExpr() {
  return getRuleContexts<ZenithParser::RelationalExprContext>();
}

ZenithParser::RelationalExprContext* ZenithParser::EqualityExprContext::relationalExpr(size_t i) {
  return getRuleContext<ZenithParser::RelationalExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::EqualityExprContext::EQ() {
  return getTokens(ZenithParser::EQ);
}

tree::TerminalNode* ZenithParser::EqualityExprContext::EQ(size_t i) {
  return getToken(ZenithParser::EQ, i);
}

std::vector<tree::TerminalNode *> ZenithParser::EqualityExprContext::NEQ() {
  return getTokens(ZenithParser::NEQ);
}

tree::TerminalNode* ZenithParser::EqualityExprContext::NEQ(size_t i) {
  return getToken(ZenithParser::NEQ, i);
}


size_t ZenithParser::EqualityExprContext::getRuleIndex() const {
  return ZenithParser::RuleEqualityExpr;
}


std::any ZenithParser::EqualityExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitEqualityExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::EqualityExprContext* ZenithParser::equalityExpr() {
  EqualityExprContext *_localctx = _tracker.createInstance<EqualityExprContext>(_ctx, getState());
  enterRule(_localctx, 66, ZenithParser::RuleEqualityExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(457);
    relationalExpr();
    setState(462);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::EQ

    || _la == ZenithParser::NEQ) {
      setState(458);
      _la = _input->LA(1);
      if (!(_la == ZenithParser::EQ

      || _la == ZenithParser::NEQ)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(459);
      relationalExpr();
      setState(464);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RelationalExprContext ------------------------------------------------------------------

ZenithParser::RelationalExprContext::RelationalExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::ShiftExprContext *> ZenithParser::RelationalExprContext::shiftExpr() {
  return getRuleContexts<ZenithParser::ShiftExprContext>();
}

ZenithParser::ShiftExprContext* ZenithParser::RelationalExprContext::shiftExpr(size_t i) {
  return getRuleContext<ZenithParser::ShiftExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::RelationalExprContext::LT() {
  return getTokens(ZenithParser::LT);
}

tree::TerminalNode* ZenithParser::RelationalExprContext::LT(size_t i) {
  return getToken(ZenithParser::LT, i);
}

std::vector<tree::TerminalNode *> ZenithParser::RelationalExprContext::LE() {
  return getTokens(ZenithParser::LE);
}

tree::TerminalNode* ZenithParser::RelationalExprContext::LE(size_t i) {
  return getToken(ZenithParser::LE, i);
}

std::vector<tree::TerminalNode *> ZenithParser::RelationalExprContext::GT() {
  return getTokens(ZenithParser::GT);
}

tree::TerminalNode* ZenithParser::RelationalExprContext::GT(size_t i) {
  return getToken(ZenithParser::GT, i);
}

std::vector<tree::TerminalNode *> ZenithParser::RelationalExprContext::GE() {
  return getTokens(ZenithParser::GE);
}

tree::TerminalNode* ZenithParser::RelationalExprContext::GE(size_t i) {
  return getToken(ZenithParser::GE, i);
}


size_t ZenithParser::RelationalExprContext::getRuleIndex() const {
  return ZenithParser::RuleRelationalExpr;
}


std::any ZenithParser::RelationalExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitRelationalExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::RelationalExprContext* ZenithParser::relationalExpr() {
  RelationalExprContext *_localctx = _tracker.createInstance<RelationalExprContext>(_ctx, getState());
  enterRule(_localctx, 68, ZenithParser::RuleRelationalExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(465);
    shiftExpr();
    setState(470);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 125829120) != 0)) {
      setState(466);
      _la = _input->LA(1);
      if (!((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 125829120) != 0))) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(467);
      shiftExpr();
      setState(472);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ShiftExprContext ------------------------------------------------------------------

ZenithParser::ShiftExprContext::ShiftExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::AdditiveExprContext *> ZenithParser::ShiftExprContext::additiveExpr() {
  return getRuleContexts<ZenithParser::AdditiveExprContext>();
}

ZenithParser::AdditiveExprContext* ZenithParser::ShiftExprContext::additiveExpr(size_t i) {
  return getRuleContext<ZenithParser::AdditiveExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::ShiftExprContext::LSHIFT() {
  return getTokens(ZenithParser::LSHIFT);
}

tree::TerminalNode* ZenithParser::ShiftExprContext::LSHIFT(size_t i) {
  return getToken(ZenithParser::LSHIFT, i);
}

std::vector<tree::TerminalNode *> ZenithParser::ShiftExprContext::RSHIFT() {
  return getTokens(ZenithParser::RSHIFT);
}

tree::TerminalNode* ZenithParser::ShiftExprContext::RSHIFT(size_t i) {
  return getToken(ZenithParser::RSHIFT, i);
}


size_t ZenithParser::ShiftExprContext::getRuleIndex() const {
  return ZenithParser::RuleShiftExpr;
}


std::any ZenithParser::ShiftExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitShiftExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::ShiftExprContext* ZenithParser::shiftExpr() {
  ShiftExprContext *_localctx = _tracker.createInstance<ShiftExprContext>(_ctx, getState());
  enterRule(_localctx, 70, ZenithParser::RuleShiftExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(473);
    additiveExpr();
    setState(478);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::LSHIFT

    || _la == ZenithParser::RSHIFT) {
      setState(474);
      _la = _input->LA(1);
      if (!(_la == ZenithParser::LSHIFT

      || _la == ZenithParser::RSHIFT)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(475);
      additiveExpr();
      setState(480);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AdditiveExprContext ------------------------------------------------------------------

ZenithParser::AdditiveExprContext::AdditiveExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::MultiplicativeExprContext *> ZenithParser::AdditiveExprContext::multiplicativeExpr() {
  return getRuleContexts<ZenithParser::MultiplicativeExprContext>();
}

ZenithParser::MultiplicativeExprContext* ZenithParser::AdditiveExprContext::multiplicativeExpr(size_t i) {
  return getRuleContext<ZenithParser::MultiplicativeExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::AdditiveExprContext::PLUS() {
  return getTokens(ZenithParser::PLUS);
}

tree::TerminalNode* ZenithParser::AdditiveExprContext::PLUS(size_t i) {
  return getToken(ZenithParser::PLUS, i);
}

std::vector<tree::TerminalNode *> ZenithParser::AdditiveExprContext::MINUS() {
  return getTokens(ZenithParser::MINUS);
}

tree::TerminalNode* ZenithParser::AdditiveExprContext::MINUS(size_t i) {
  return getToken(ZenithParser::MINUS, i);
}


size_t ZenithParser::AdditiveExprContext::getRuleIndex() const {
  return ZenithParser::RuleAdditiveExpr;
}


std::any ZenithParser::AdditiveExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitAdditiveExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::AdditiveExprContext* ZenithParser::additiveExpr() {
  AdditiveExprContext *_localctx = _tracker.createInstance<AdditiveExprContext>(_ctx, getState());
  enterRule(_localctx, 72, ZenithParser::RuleAdditiveExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(481);
    multiplicativeExpr();
    setState(486);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == ZenithParser::PLUS

    || _la == ZenithParser::MINUS) {
      setState(482);
      _la = _input->LA(1);
      if (!(_la == ZenithParser::PLUS

      || _la == ZenithParser::MINUS)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(483);
      multiplicativeExpr();
      setState(488);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiplicativeExprContext ------------------------------------------------------------------

ZenithParser::MultiplicativeExprContext::MultiplicativeExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<ZenithParser::PowerExprContext *> ZenithParser::MultiplicativeExprContext::powerExpr() {
  return getRuleContexts<ZenithParser::PowerExprContext>();
}

ZenithParser::PowerExprContext* ZenithParser::MultiplicativeExprContext::powerExpr(size_t i) {
  return getRuleContext<ZenithParser::PowerExprContext>(i);
}

std::vector<tree::TerminalNode *> ZenithParser::MultiplicativeExprContext::STAR() {
  return getTokens(ZenithParser::STAR);
}

tree::TerminalNode* ZenithParser::MultiplicativeExprContext::STAR(size_t i) {
  return getToken(ZenithParser::STAR, i);
}

std::vector<tree::TerminalNode *> ZenithParser::MultiplicativeExprContext::DIV() {
  return getTokens(ZenithParser::DIV);
}

tree::TerminalNode* ZenithParser::MultiplicativeExprContext::DIV(size_t i) {
  return getToken(ZenithParser::DIV, i);
}

std::vector<tree::TerminalNode *> ZenithParser::MultiplicativeExprContext::MOD() {
  return getTokens(ZenithParser::MOD);
}

tree::TerminalNode* ZenithParser::MultiplicativeExprContext::MOD(size_t i) {
  return getToken(ZenithParser::MOD, i);
}


size_t ZenithParser::MultiplicativeExprContext::getRuleIndex() const {
  return ZenithParser::RuleMultiplicativeExpr;
}


std::any ZenithParser::MultiplicativeExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitMultiplicativeExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::MultiplicativeExprContext* ZenithParser::multiplicativeExpr() {
  MultiplicativeExprContext *_localctx = _tracker.createInstance<MultiplicativeExprContext>(_ctx, getState());
  enterRule(_localctx, 74, ZenithParser::RuleMultiplicativeExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(489);
    powerExpr();
    setState(494);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 917504) != 0)) {
      setState(490);
      _la = _input->LA(1);
      if (!((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 917504) != 0))) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(491);
      powerExpr();
      setState(496);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PowerExprContext ------------------------------------------------------------------

ZenithParser::PowerExprContext::PowerExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::UnaryExprContext* ZenithParser::PowerExprContext::unaryExpr() {
  return getRuleContext<ZenithParser::UnaryExprContext>(0);
}

tree::TerminalNode* ZenithParser::PowerExprContext::POW() {
  return getToken(ZenithParser::POW, 0);
}

ZenithParser::PowerExprContext* ZenithParser::PowerExprContext::powerExpr() {
  return getRuleContext<ZenithParser::PowerExprContext>(0);
}


size_t ZenithParser::PowerExprContext::getRuleIndex() const {
  return ZenithParser::RulePowerExpr;
}


std::any ZenithParser::PowerExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitPowerExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::PowerExprContext* ZenithParser::powerExpr() {
  PowerExprContext *_localctx = _tracker.createInstance<PowerExprContext>(_ctx, getState());
  enterRule(_localctx, 76, ZenithParser::RulePowerExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(497);
    unaryExpr();
    setState(500);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == ZenithParser::POW) {
      setState(498);
      match(ZenithParser::POW);
      setState(499);
      powerExpr();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryExprContext ------------------------------------------------------------------

ZenithParser::UnaryExprContext::UnaryExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::UnaryExprContext* ZenithParser::UnaryExprContext::unaryExpr() {
  return getRuleContext<ZenithParser::UnaryExprContext>(0);
}

tree::TerminalNode* ZenithParser::UnaryExprContext::NOT() {
  return getToken(ZenithParser::NOT, 0);
}

tree::TerminalNode* ZenithParser::UnaryExprContext::MINUS() {
  return getToken(ZenithParser::MINUS, 0);
}

tree::TerminalNode* ZenithParser::UnaryExprContext::TILDE() {
  return getToken(ZenithParser::TILDE, 0);
}

tree::TerminalNode* ZenithParser::UnaryExprContext::AMPERSAND() {
  return getToken(ZenithParser::AMPERSAND, 0);
}

ZenithParser::CallExprContext* ZenithParser::UnaryExprContext::callExpr() {
  return getRuleContext<ZenithParser::CallExprContext>(0);
}


size_t ZenithParser::UnaryExprContext::getRuleIndex() const {
  return ZenithParser::RuleUnaryExpr;
}


std::any ZenithParser::UnaryExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitUnaryExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::UnaryExprContext* ZenithParser::unaryExpr() {
  UnaryExprContext *_localctx = _tracker.createInstance<UnaryExprContext>(_ctx, getState());
  enterRule(_localctx, 78, ZenithParser::RuleUnaryExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(505);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ZenithParser::MINUS:
      case ZenithParser::NOT:
      case ZenithParser::AMPERSAND:
      case ZenithParser::TILDE: {
        enterOuterAlt(_localctx, 1);
        setState(502);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 10200612864) != 0))) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(503);
        unaryExpr();
        break;
      }

      case ZenithParser::TRUE:
      case ZenithParser::FALSE:
      case ZenithParser::LPAREN:
      case ZenithParser::INTEGER:
      case ZenithParser::FLOAT:
      case ZenithParser::STRING:
      case ZenithParser::IDENTIFIER: {
        enterOuterAlt(_localctx, 2);
        setState(504);
        callExpr();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CallExprContext ------------------------------------------------------------------

ZenithParser::CallExprContext::CallExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

ZenithParser::PrimaryExprContext* ZenithParser::CallExprContext::primaryExpr() {
  return getRuleContext<ZenithParser::PrimaryExprContext>(0);
}

std::vector<ZenithParser::CallSuffixContext *> ZenithParser::CallExprContext::callSuffix() {
  return getRuleContexts<ZenithParser::CallSuffixContext>();
}

ZenithParser::CallSuffixContext* ZenithParser::CallExprContext::callSuffix(size_t i) {
  return getRuleContext<ZenithParser::CallSuffixContext>(i);
}


size_t ZenithParser::CallExprContext::getRuleIndex() const {
  return ZenithParser::RuleCallExpr;
}


std::any ZenithParser::CallExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitCallExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::CallExprContext* ZenithParser::callExpr() {
  CallExprContext *_localctx = _tracker.createInstance<CallExprContext>(_ctx, getState());
  enterRule(_localctx, 80, ZenithParser::RuleCallExpr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(507);
    primaryExpr();
    setState(511);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 36352603193344) != 0)) {
      setState(508);
      callSuffix();
      setState(513);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CallSuffixContext ------------------------------------------------------------------

ZenithParser::CallSuffixContext::CallSuffixContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::CallSuffixContext::LBRACKET() {
  return getToken(ZenithParser::LBRACKET, 0);
}

std::vector<ZenithParser::ExpressionContext *> ZenithParser::CallSuffixContext::expression() {
  return getRuleContexts<ZenithParser::ExpressionContext>();
}

ZenithParser::ExpressionContext* ZenithParser::CallSuffixContext::expression(size_t i) {
  return getRuleContext<ZenithParser::ExpressionContext>(i);
}

tree::TerminalNode* ZenithParser::CallSuffixContext::RBRACKET() {
  return getToken(ZenithParser::RBRACKET, 0);
}

tree::TerminalNode* ZenithParser::CallSuffixContext::DOT() {
  return getToken(ZenithParser::DOT, 0);
}

tree::TerminalNode* ZenithParser::CallSuffixContext::IDENTIFIER() {
  return getToken(ZenithParser::IDENTIFIER, 0);
}

tree::TerminalNode* ZenithParser::CallSuffixContext::LPAREN() {
  return getToken(ZenithParser::LPAREN, 0);
}

tree::TerminalNode* ZenithParser::CallSuffixContext::RPAREN() {
  return getToken(ZenithParser::RPAREN, 0);
}

std::vector<tree::TerminalNode *> ZenithParser::CallSuffixContext::COMMA() {
  return getTokens(ZenithParser::COMMA);
}

tree::TerminalNode* ZenithParser::CallSuffixContext::COMMA(size_t i) {
  return getToken(ZenithParser::COMMA, i);
}


size_t ZenithParser::CallSuffixContext::getRuleIndex() const {
  return ZenithParser::RuleCallSuffix;
}


std::any ZenithParser::CallSuffixContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitCallSuffix(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::CallSuffixContext* ZenithParser::callSuffix() {
  CallSuffixContext *_localctx = _tracker.createInstance<CallSuffixContext>(_ctx, getState());
  enterRule(_localctx, 82, ZenithParser::RuleCallSuffix);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(532);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ZenithParser::LBRACKET: {
        enterOuterAlt(_localctx, 1);
        setState(514);
        match(ZenithParser::LBRACKET);
        setState(515);
        expression();
        setState(516);
        match(ZenithParser::RBRACKET);
        break;
      }

      case ZenithParser::DOT: {
        enterOuterAlt(_localctx, 2);
        setState(518);
        match(ZenithParser::DOT);
        setState(519);
        match(ZenithParser::IDENTIFIER);
        break;
      }

      case ZenithParser::LPAREN: {
        enterOuterAlt(_localctx, 3);
        setState(520);
        match(ZenithParser::LPAREN);
        setState(529);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 8444328221415424) != 0)) {
          setState(521);
          expression();
          setState(526);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while (_la == ZenithParser::COMMA) {
            setState(522);
            match(ZenithParser::COMMA);
            setState(523);
            expression();
            setState(528);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
        }
        setState(531);
        match(ZenithParser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrimaryExprContext ------------------------------------------------------------------

ZenithParser::PrimaryExprContext::PrimaryExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* ZenithParser::PrimaryExprContext::INTEGER() {
  return getToken(ZenithParser::INTEGER, 0);
}

tree::TerminalNode* ZenithParser::PrimaryExprContext::FLOAT() {
  return getToken(ZenithParser::FLOAT, 0);
}

tree::TerminalNode* ZenithParser::PrimaryExprContext::STRING() {
  return getToken(ZenithParser::STRING, 0);
}

tree::TerminalNode* ZenithParser::PrimaryExprContext::TRUE() {
  return getToken(ZenithParser::TRUE, 0);
}

tree::TerminalNode* ZenithParser::PrimaryExprContext::FALSE() {
  return getToken(ZenithParser::FALSE, 0);
}

tree::TerminalNode* ZenithParser::PrimaryExprContext::IDENTIFIER() {
  return getToken(ZenithParser::IDENTIFIER, 0);
}

tree::TerminalNode* ZenithParser::PrimaryExprContext::LPAREN() {
  return getToken(ZenithParser::LPAREN, 0);
}

ZenithParser::ExpressionContext* ZenithParser::PrimaryExprContext::expression() {
  return getRuleContext<ZenithParser::ExpressionContext>(0);
}

tree::TerminalNode* ZenithParser::PrimaryExprContext::RPAREN() {
  return getToken(ZenithParser::RPAREN, 0);
}


size_t ZenithParser::PrimaryExprContext::getRuleIndex() const {
  return ZenithParser::RulePrimaryExpr;
}


std::any ZenithParser::PrimaryExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<ZenithParserVisitor*>(visitor))
    return parserVisitor->visitPrimaryExpr(this);
  else
    return visitor->visitChildren(this);
}

ZenithParser::PrimaryExprContext* ZenithParser::primaryExpr() {
  PrimaryExprContext *_localctx = _tracker.createInstance<PrimaryExprContext>(_ctx, getState());
  enterRule(_localctx, 84, ZenithParser::RulePrimaryExpr);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(544);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case ZenithParser::INTEGER: {
        enterOuterAlt(_localctx, 1);
        setState(534);
        match(ZenithParser::INTEGER);
        break;
      }

      case ZenithParser::FLOAT: {
        enterOuterAlt(_localctx, 2);
        setState(535);
        match(ZenithParser::FLOAT);
        break;
      }

      case ZenithParser::STRING: {
        enterOuterAlt(_localctx, 3);
        setState(536);
        match(ZenithParser::STRING);
        break;
      }

      case ZenithParser::TRUE: {
        enterOuterAlt(_localctx, 4);
        setState(537);
        match(ZenithParser::TRUE);
        break;
      }

      case ZenithParser::FALSE: {
        enterOuterAlt(_localctx, 5);
        setState(538);
        match(ZenithParser::FALSE);
        break;
      }

      case ZenithParser::IDENTIFIER: {
        enterOuterAlt(_localctx, 6);
        setState(539);
        match(ZenithParser::IDENTIFIER);
        break;
      }

      case ZenithParser::LPAREN: {
        enterOuterAlt(_localctx, 7);
        setState(540);
        match(ZenithParser::LPAREN);
        setState(541);
        expression();
        setState(542);
        match(ZenithParser::RPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

void ZenithParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  zenithparserParserInitialize();
#else
  ::antlr4::internal::call_once(zenithparserParserOnceFlag, zenithparserParserInitialize);
#endif
}
