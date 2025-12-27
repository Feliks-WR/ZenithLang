use std::env;
use std::fs;
use std::fmt;
use std::ffi::CString;
use std::os::raw::c_char;

// --- FFI Bindings to C++ MLIR Bridge ---

#[link(name = "ZenithDialect")]
unsafe extern "C" {
    fn zenith_create_module() -> *mut std::ffi::c_void;
    fn zenith_destroy_module(module: *mut std::ffi::c_void);
    fn zenith_emit_assign(module: *mut std::ffi::c_void, name: *const c_char, value: i64);
    fn zenith_emit_array(module: *mut std::ffi::c_void, name: *const c_char, elements: *const i64, count: usize);
    fn zenith_emit_string(module: *mut std::ffi::c_void, name: *const c_char, value: *const c_char);
    fn zenith_emit_print(module: *mut std::ffi::c_void, var_name: *const c_char);
    fn zenith_finalize(module: *mut std::ffi::c_void);
    fn zenith_dump(module: *mut std::ffi::c_void);
    fn zenith_execute(module: *mut std::ffi::c_void) -> i32;
}

// --- Lexer ---

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Identifier(String),
    IntegerLiteral(i64),
    StringLiteral(String),
    Assign,
    LBracket,
    RBracket,
    Comma,
    Print,
    Newline,
    EOF,
    ERROR
}

// --- High-Level IR (HIR) ---

#[derive(Debug, Clone)]
enum Expression {
    Integer(i64),
    Array(Vec<i64>),
    String(String),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expression::Integer(value) => write!(f, "{}", value),
            Expression::Array(values) => write!(f, "[{}]", values.iter().map(|v| v.to_string()).collect::<Vec<String>>().join(", ")),
            Expression::String(value) => write!(f, "\"{}\"", value),
        }
    }
}

#[derive(Debug, Clone)]
enum Statement {
    Assignment { name: String, value: Expression },
    Print(String),
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Statement::Assignment { name, value } => write!(f, "{} = {}", name, value),
            Statement::Print(name) => write!(f, "print {}", name),
        }
    }
}

#[derive(Debug, Clone)]
struct Program {
    statements: Vec<Statement>,
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for stmt in &self.statements {
            write!(f, "{}", stmt)?;
        }
        Ok(())
    }
}

// --- Lexer ---

struct Lexer<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Lexer { input, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        let c = match self.advance() {
            Some(c) => c,
            None => return Token::EOF,
        };

        match c {
            '\n' => Token::Newline,
            '=' => Token::Assign,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            ',' => Token::Comma,
            '"' => {
                let mut s = String::new();
                while let Some(c) = self.advance() {
                    if c == '"' {
                        break;
                    }
                    s.push(c);
                }
                Token::StringLiteral(s)
            }
            c if c.is_ascii_digit() => {
                let mut s = String::new();
                s.push(c);
                while let Some(nc) = self.peek() {
                    if nc.is_ascii_digit() {
                        s.push(self.advance().unwrap());
                    } else {
                        break;
                    }
                }
                Token::IntegerLiteral(s.parse().unwrap())
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut s = String::new();
                s.push(c);
                while let Some(nc) = self.peek() {
                    if nc.is_alphanumeric() || nc == '_' {
                        s.push(self.advance().unwrap());
                    } else {
                        break;
                    }
                }
                match s.as_str() {
                    "print" => Token::Print,
                    _ => Token::Identifier(s),
                }
            }
            _ => Token::ERROR
        }
    }
}

// --- Parser ---

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    errors: Vec<String>,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            pos: 0,
            errors: vec![],
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let res = self.tokens.get(self.pos).cloned();
        if res.is_some() {
            self.pos += 1;
        }
        res
    }

    fn error(&mut self, msg: String) {
        self.errors.push(msg);
    }

    fn expect(&mut self, expected: Token) {
        if let Some(tok) = self.advance() {
            if tok != expected {
                self.error(format!("Expected {:?}, found {:?}", expected, tok));
            }
        } else {
            self.error(format!("Expected {:?}, found EOF", expected));
        }
    }

    fn parse(&mut self) -> Program {
        let mut statements = vec![];
        while self.pos < self.tokens.len() {
            match self.peek() {
                Some(Token::Identifier(_)) => {
                    if let Some(stmt) = self.parse_assignment() {
                        statements.push(stmt);
                    }
                }
                Some(Token::Print) => {
                    if let Some(stmt) = self.parse_print() {
                        statements.push(stmt);
                    }
                }
                Some(Token::Newline) => {
                    self.advance();
                }
                Some(tok) => {
                    self.error(format!("Unexpected token: {:?}", tok));
                    self.advance();
                }
                None => break,
            }
        }
        Program { statements }
    }

    fn parse_expression(&mut self) -> Option<Expression> {
        match self.advance()?.clone() {
            Token::IntegerLiteral(i) => Some(Expression::Integer(i)),
            Token::LBracket => {
                let mut elements = vec![];
                while let Some(tok) = self.peek() {
                    if tok == &Token::RBracket {
                        break;
                    }
                    if let Some(Token::IntegerLiteral(i)) = self.advance() {
                        elements.push(i);
                    } else {
                        self.error("Expected integer in array".to_string());
                        break;
                    }
                    if self.peek() == Some(&Token::Comma) {
                        self.advance();
                    }
                }
                self.expect(Token::RBracket);
                Some(Expression::Array(elements))
            }
            Token::StringLiteral(s) => Some(Expression::String(s)),
            _ => {
                self.error("Expected value".to_string());
                None
            }
        }
    }

    fn parse_assignment(&mut self) -> Option<Statement> {
        let name = if let Some(Token::Identifier(name)) = self.advance() {
            name.clone()
        } else {
            return None;
        };

        self.expect(Token::Assign);
        let value = self.parse_expression()?;
        Some(Statement::Assignment { name, value })
    }

    fn parse_print(&mut self) -> Option<Statement> {
        self.expect(Token::Print);
        if let Some(Token::Identifier(name)) = self.advance() {
            Some(Statement::Print(name.clone()))
        } else {
            self.error("Expected identifier after print".to_string());
            None
        }
    }
}

// --- Reversible High-Level IR ---

/// High-level IR that preserves all information for reversible transformation
#[derive(Debug, Clone, PartialEq)]
struct IR {
    statements: Vec<IRStatement>,
}

#[derive(Debug, Clone, PartialEq)]
enum IRStatement {
    /// Variable assignment: name = expression
    Assign {
        variable: IRVariable,
        expression: IRExpression,
    },
    /// Print statement: print variable
    Print {
        variable: IRVariable,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct IRVariable {
    name: String,
}

#[derive(Debug, Clone, PartialEq)]
enum IRExpression {
    /// Integer literal
    IntLiteral(i64),
    /// Array literal [elem1, elem2, ...]
    ArrayLiteral(Vec<i64>),
    /// String literal "..."
    StringLiteral(String),
}

// --- MLIR Generator ---

struct MLIRGenerator {
    module: *mut std::ffi::c_void,
}

impl MLIRGenerator {
    fn new() -> Self {
        unsafe {
            MLIRGenerator {
                module: zenith_create_module(),
            }
        }
    }

    fn generate(&self, ir: &IR) {
        for stmt in &ir.statements {
            match stmt {
                IRStatement::Assign { variable, expression } => {
                    let name = CString::new(variable.name.as_str()).unwrap();
                    match expression {
                        IRExpression::IntLiteral(value) => {
                            unsafe {
                                zenith_emit_assign(self.module, name.as_ptr(), *value);
                            }
                        }
                        IRExpression::ArrayLiteral(elements) => {
                            unsafe {
                                zenith_emit_array(
                                    self.module,
                                    name.as_ptr(),
                                    elements.as_ptr(),
                                    elements.len()
                                );
                            }
                        }
                        IRExpression::StringLiteral(value) => {
                            let value_cstr = CString::new(value.as_str()).unwrap();
                            unsafe {
                                zenith_emit_string(self.module, name.as_ptr(), value_cstr.as_ptr());
                            }
                        }
                    }
                }
                IRStatement::Print { variable } => {
                    let name = CString::new(variable.name.as_str()).unwrap();
                    unsafe {
                        zenith_emit_print(self.module, name.as_ptr());
                    }
                }
            }
        }
    }

    fn finalize(&self) {
        unsafe {
            zenith_finalize(self.module);
        }
    }

    fn dump(&self) {
        unsafe {
            zenith_dump(self.module);
        }
    }

    fn execute(&self) -> i32 {
        unsafe {
            zenith_execute(self.module)
        }
    }
}

impl Drop for MLIRGenerator {
    fn drop(&mut self) {
        unsafe {
            zenith_destroy_module(self.module);
        }
    }
}

// --- S-Expression Formatting ---

impl fmt::Display for IR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(module")?;
        for stmt in &self.statements {
            write!(f, "\n  {}", stmt)?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for IRStatement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IRStatement::Assign { variable, expression } => {
                write!(f, "(assign {} {})", variable, expression)
            }
            IRStatement::Print { variable } => {
                write!(f, "(print {})", variable)
            }
        }
    }
}

impl fmt::Display for IRVariable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Display for IRExpression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IRExpression::IntLiteral(i) => write!(f, "(int {})", i),
            IRExpression::ArrayLiteral(elements) => {
                write!(f, "(array")?;
                for elem in elements {
                    write!(f, " {}", elem)?;
                }
                write!(f, ")")
            }
            IRExpression::StringLiteral(s) => write!(f, "(string \"{}\")", s),
        }
    }
}

/// Builder to convert Program (HIR) to IR
struct IRBuilder;

impl IRBuilder {
    fn build(program: Program) -> IR {
        let statements = program
            .statements
            .into_iter()
            .map(|stmt| Self::convert_statement(stmt))
            .collect();

        IR { statements }
    }

    fn convert_statement(stmt: Statement) -> IRStatement {
        match stmt {
            Statement::Assignment { name, value } => IRStatement::Assign {
                variable: IRVariable { name },
                expression: Self::convert_expression(value),
            },
            Statement::Print(name) => IRStatement::Print {
                variable: IRVariable { name },
            },
        }
    }

    fn convert_expression(expr: Expression) -> IRExpression {
        match expr {
            Expression::Integer(i) => IRExpression::IntLiteral(i),
            Expression::Array(elements) => IRExpression::ArrayLiteral(elements),
            Expression::String(s) => IRExpression::StringLiteral(s),
        }
    }
}

/// Printer to convert IR back to source code (reversible)
struct IRPrinter;

impl IRPrinter {
    fn print(ir: &IR) -> String {
        let mut output = String::new();

        for stmt in &ir.statements {
            output.push_str(&Self::print_statement(stmt));
            output.push('\n');
        }

        output
    }

    fn print_statement(stmt: &IRStatement) -> String {
        match stmt {
            IRStatement::Assign { variable, expression } => {
                format!("{} = {}", variable.name, Self::print_expression(expression))
            }
            IRStatement::Print { variable } => {
                format!("print {}", variable.name)
            }
        }
    }

    fn print_expression(expr: &IRExpression) -> String {
        match expr {
            IRExpression::IntLiteral(i) => i.to_string(),
            IRExpression::ArrayLiteral(elements) => {
                let elements_str = elements
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{}]", elements_str)
            }
            IRExpression::StringLiteral(s) => format!("\"{}\"", s),
        }
    }

    /// Convert IR back to Program (HIR)
    fn ir_to_program(ir: &IR) -> Program {
        let statements = ir
            .statements
            .iter()
            .map(|stmt| Self::convert_ir_statement(stmt))
            .collect();

        Program { statements }
    }

    fn convert_ir_statement(stmt: &IRStatement) -> Statement {
        match stmt {
            IRStatement::Assign { variable, expression } => Statement::Assignment {
                name: variable.name.clone(),
                value: Self::convert_ir_expression(expression),
            },
            IRStatement::Print { variable } => Statement::Print(variable.name.clone()),
        }
    }

    fn convert_ir_expression(expr: &IRExpression) -> Expression {
        match expr {
            IRExpression::IntLiteral(i) => Expression::Integer(*i),
            IRExpression::ArrayLiteral(elements) => Expression::Array(elements.clone()),
            IRExpression::StringLiteral(s) => Expression::String(s.clone()),
        }
    }
}


// --- Main ---

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <input_file>",
            args.first().expect("A program name must be given as first argument")
        );
        return;
    }

    let input = fs::read_to_string(&args[1])
        .expect(format!("Second argument '{}' must be a file path", args[1]).as_str());
    let mut lexer = Lexer::new(&input);
    let mut tokens = vec![];
    loop {
        let tok = lexer.next_token();
        if tok == Token::EOF { break; }
        tokens.push(tok);
    }

    let mut parser = Parser::new(tokens);
    let program = parser.parse();

    if !parser.errors.is_empty() {
        for err in &parser.errors {
            eprintln!("Parse Error: {}", err);
        }
        return;
    }

    // Convert to HIR
    let ir = IRBuilder::build(program.clone());

    // println!("\n--- HIR (Debug) ---");
    // println!("{:#?}", hir);

    println!("\n--- HIR (S-Expression) ---");
    println!("{}", ir);

    // Convert HIR back to source code (demonstrating reversibility)
    println!("\n--- Reconstructed Source Code ---");
    let reconstructed_code = IRPrinter::print(&ir);
    print!("{}", reconstructed_code);

    // Verify reversibility: HIR -> Program -> HIR should be identical
    println!("\n--- Reversibility Check ---");
    let reconstructed_program = IRPrinter::ir_to_program(&ir);
    let ir2 = IRBuilder::build(reconstructed_program);

    if ir == ir2 {
        println!("✓ Transformation is reversible: Source code <-> HIR");
    } else {
        eprintln!("✗ Transformation is NOT reversible");
        // Show diff
        let str = program.to_string();
        let diff = diff::lines(&str, &reconstructed_code);
        for change in diff {
            match change {
                diff::Result::Left(l) => eprintln!("-{}", l),
                diff::Result::Right(r) => eprintln!("+{}", r),
                diff::Result::Both(_, _) => {}
            }
        }
    }

    println!("\n--- Generating MLIR via C++ Bridge ---");
    let generator = MLIRGenerator::new();
    generator.generate(&ir);
    generator.finalize();
    generator.dump();

    println!("\n--- Executing MLIR ---");
    let exit_code = generator.execute();
    if exit_code != 0 {
        eprintln!("Execution failed with code {}", exit_code);
    }
}
