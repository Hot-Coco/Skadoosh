//! Built-in tools that execute locally — no subprocess, no external API.
//!
//! Each tool is a pure-Rust function dispatched by name in [`execute`].
//! [`definitions`] returns the function schemas so the LLM knows about them
//! without a user-provided tools JSON file.
//!
//! # Tools
//!
//! | Tool         | Description                              |
//! |--------------|------------------------------------------|
//! | `calculator` | Safe math evaluation (no `eval`)         |
//! | `datetime`   | Current date / time in several formats   |
//! | `remember`   | Save a key-value note to agent memory     |
//! | `recall`     | Search saved memories by keyword          |

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::Result;
use crate::llm::client::Tool;
use crate::memory::MemoryStore;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Returns `true` when `name` is a recognised built-in tool.
pub fn is_builtin(name: &str) -> bool {
    matches!(name, "calculator" | "datetime" | "remember" | "recall")
}

/// Execute a built-in tool that doesn't need memory access.
///
/// Returns `None` when `name` is not a recognised built-in — the caller
/// should fall through to the shell executor.  `args_json` is the raw JSON
/// arguments string the model emitted (e.g. `{"expression": "2+2"}`).
pub fn execute(name: &str, args_json: &str) -> Option<Result<String>> {
    let args: Value = match serde_json::from_str(args_json) {
        Ok(v) => v,
        Err(e) => {
            return Some(Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                "invalid tool arguments: {e}"
            ))))
        }
    };
    match name {
        "calculator" => Some(calc(args["expression"].as_str().unwrap_or(""))),
        "datetime" => Some(now(args["format"].as_str())),
        "remember" | "recall" => {
            // Memory-backed — the caller must route these through
            // execute_with_memory when a MemoryStore is available.
            None
        }
        _ => None,
    }
}

/// Execute a memory-backed built-in tool (`remember`, `recall`).
///
/// Accepts a `Mutex<MemoryStore>` so the caller can lock it for writes
/// (`remember`) or reads (`recall`).  Returns `None` for non-memory tools.
pub fn execute_with_memory(
    name: &str,
    args_json: &str,
    store: &Mutex<MemoryStore>,
) -> Option<Result<String>> {
    let args: Value = match serde_json::from_str(args_json) {
        Ok(v) => v,
        Err(e) => {
            return Some(Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                "invalid tool arguments: {e}"
            ))))
        }
    };
    match name {
        "remember" => Some(rem(
            args["key"].as_str().unwrap_or(""),
            args["value"].as_str().unwrap_or(""),
            store,
        )),
        "recall" => Some(rec(args["query"].as_str().unwrap_or(""), store)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tool definitions (function schemas for the LLM)
// ---------------------------------------------------------------------------

/// Tool definitions for all built-in tools.  Merge these with any
/// user-supplied tools file so the LLM always knows about built-ins.
pub fn definitions() -> Vec<Tool> {
    vec![
        Tool::function(
            "calculator",
            "Evaluate a mathematical expression. Supports +, -, *, /, %, ^ (power), \
             sqrt, abs, sin, cos, tan, ln, log, floor, ceil, round, pow, and parentheses. \
             Use for arithmetic, percentages, unit conversions, and simple calculations.",
            json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression, e.g. '2+3*4' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }),
        ),
        Tool::function(
            "datetime",
            "Get the current date and time.",
            json!({
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["iso", "date", "time", "unix", "friendly"],
                        "description": "Output format. 'iso' = ISO 8601, 'date' = YYYY-MM-DD, \
                            'time' = HH:MM:SS, 'unix' = seconds since epoch, \
                            'friendly' = human readable"
                    }
                }
            }),
        ),
        Tool::function(
            "remember",
            "Save a piece of information so the agent can recall it in future \
             conversations. Use this when the user asks you to remember something — \
             their name, a preference, a fact they shared.",
            json!({
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short label, e.g. 'user-name' or 'favorite-color'"
                    },
                    "value": {
                        "type": "string",
                        "description": "The information to remember"
                    }
                },
                "required": ["key", "value"]
            }),
        ),
        Tool::function(
            "recall",
            "Search the agent's saved memories. Use this when the user asks about \
             something they previously told you to remember, or when you need context \
             from past conversations.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or phrase to search for in saved memories"
                    }
                }
            }),
        ),
    ]
}

// ---------------------------------------------------------------------------
// calculator
// ---------------------------------------------------------------------------

/// Safe math evaluator: tokenise → shunting-yard → RPN → evaluate.
fn calc(expression: &str) -> Result<String> {
    let tokens = tokenize(expression)?;
    let rpn = shunt(&tokens)?;
    let result = eval_rpn(&rpn)?;
    // Print integer-looking results without a trailing ".0".
    if (result - result.round()).abs() < 1e-10 {
        Ok(format!("{}", result.round() as i64))
    } else {
        Ok(format!("{result:.10}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Op(char),
    LParen,
    RParen,
    Func(String),
}

fn tokenize(expr: &str) -> Result<Vec<Tok>> {
    let mut tokens = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '0'..='9' | '.' => {
                let mut num = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        num.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let n: f64 = num.parse().map_err(|_| {
                    crate::error::SkadooshError::Other(anyhow::anyhow!(
                        "calculator: invalid number '{num}'"
                    ))
                })?;
                tokens.push(Tok::Num(n));
            }
            '+' | '-' => {
                // Unary +/- at start, after `(`, or after another operator.
                let is_unary = tokens.is_empty()
                    || tokens
                        .last()
                        .is_some_and(|t| matches!(t, Tok::LParen | Tok::Op(_)));
                chars.next();
                if is_unary {
                    if c == '-' {
                        // Peek ahead: if a number follows, negate it directly
                        // so `3*-2` = -6 rather than (3*0)-2 = -2.
                        let mut num = String::new();
                        while let Some(&d) = chars.peek() {
                            if d.is_ascii_digit() || d == '.' {
                                num.push(d);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        if !num.is_empty() {
                            let n: f64 = num.parse().map_err(|_| {
                                crate::error::SkadooshError::Other(anyhow::anyhow!(
                                    "calculator: invalid number '{num}'"
                                ))
                            })?;
                            tokens.push(Tok::Num(-n));
                        } else {
                            // `-(expr)` — push `0 -`.
                            tokens.push(Tok::Num(0.0));
                            tokens.push(Tok::Op('-'));
                        }
                    }
                    // Unary `+` is a no-op.
                } else {
                    tokens.push(Tok::Op(c));
                }
            }
            '*' | '/' | '%' | '^' => {
                tokens.push(Tok::Op(c));
                chars.next();
            }
            '(' => {
                tokens.push(Tok::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Tok::RParen);
                chars.next();
            }
            'a'..='z' | 'A'..='Z' => {
                let mut func = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_alphabetic() {
                        func.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let lower = func.to_lowercase();
                match lower.as_str() {
                    "sqrt" | "abs" | "sin" | "cos" | "tan" | "ln" | "log" | "floor" | "ceil"
                    | "round" | "pow" => {
                        tokens.push(Tok::Func(lower));
                    }
                    "pi" => tokens.push(Tok::Num(std::f64::consts::PI)),
                    "e" => tokens.push(Tok::Num(std::f64::consts::E)),
                    _ => {
                        return Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                            "calculator: unknown function or constant '{func}'"
                        )));
                    }
                }
            }
            ',' => {
                // Function argument separator — ignore (just skip it).
                chars.next();
            }
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            _ => {
                return Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                    "calculator: unexpected character '{c}'"
                )));
            }
        }
    }
    Ok(tokens)
}

fn shunt(tokens: &[Tok]) -> Result<Vec<Tok>> {
    let mut output: Vec<Tok> = Vec::new();
    let mut ops: Vec<Tok> = Vec::new();

    for tok in tokens {
        match tok {
            Tok::Num(_) => output.push(tok.clone()),
            Tok::Func(_) => ops.push(tok.clone()),
            Tok::Op(c) => {
                while let Some(top) = ops.last() {
                    let top_prec = precedence(top);
                    let cur_prec = precedence(tok);
                    // ^ is right-associative (lower prec on stack doesn't pop)
                    if *c == '^' && top_prec <= cur_prec {
                        break;
                    }
                    if top_prec < cur_prec {
                        break;
                    }
                    output.push(ops.pop().unwrap());
                }
                ops.push(tok.clone());
            }
            Tok::LParen => ops.push(tok.clone()),
            Tok::RParen => {
                while let Some(top) = ops.last() {
                    if matches!(top, Tok::LParen) {
                        break;
                    }
                    output.push(ops.pop().unwrap());
                }
                if ops.pop().is_none() {
                    return Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                        "calculator: mismatched parentheses"
                    )));
                }
                // If a function was immediately before the LParen, pop it now
                // (shunting-yard for function calls).
                if let Some(Tok::Func(_)) = ops.last() {
                    output.push(ops.pop().unwrap());
                }
            }
        }
    }
    while let Some(op) = ops.pop() {
        if matches!(op, Tok::LParen) {
            return Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                "calculator: mismatched parentheses"
            )));
        }
        output.push(op);
    }
    Ok(output)
}

fn precedence(tok: &Tok) -> u8 {
    match tok {
        Tok::Op('+') | Tok::Op('-') => 2,
        Tok::Op('*') | Tok::Op('/') | Tok::Op('%') => 3,
        Tok::Op('^') => 4,
        Tok::Func(_) => 5,
        _ => 0,
    }
}

fn eval_rpn(rpn: &[Tok]) -> Result<f64> {
    let mut stack: Vec<f64> = Vec::new();
    for tok in rpn {
        match tok {
            Tok::Num(n) => stack.push(*n),
            Tok::Op(op) => {
                let b = stack.pop().ok_or_else(|| anyhow_err("missing operand"))?;
                let a = stack.pop().ok_or_else(|| anyhow_err("missing operand"))?;
                let r = match op {
                    '+' => a + b,
                    '-' => a - b,
                    '*' => a * b,
                    '/' => {
                        if b == 0.0 {
                            return Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                                "calculator: division by zero"
                            )));
                        }
                        a / b
                    }
                    '%' => a % b,
                    '^' => a.powf(b),
                    _ => {
                        return Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                            "calculator: unknown operator '{op}'"
                        )))
                    }
                };
                stack.push(r);
            }
            Tok::Func(f) => {
                let a = stack
                    .pop()
                    .ok_or_else(|| anyhow_err("missing function argument"))?;
                let r = match f.as_str() {
                    "sqrt" => a.sqrt(),
                    "abs" => a.abs(),
                    "sin" => a.sin(),
                    "cos" => a.cos(),
                    "tan" => a.tan(),
                    "ln" => a.ln(),
                    "log" => a.log10(),
                    "floor" => a.floor(),
                    "ceil" => a.ceil(),
                    "round" => a.round(),
                    "pow" => {
                        let b = stack
                            .pop()
                            .ok_or_else(|| anyhow_err("pow needs two arguments"))?;
                        b.powf(a)
                    }
                    _ => {
                        return Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                            "calculator: unknown function '{f}'"
                        )))
                    }
                };
                stack.push(r);
            }
            _ => {}
        }
    }
    match stack.len() {
        1 => Ok(stack[0]),
        0 => Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
            "calculator: empty expression"
        ))),
        n => Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
            "calculator: {n} values left on stack — missing operator?"
        ))),
    }
}

fn anyhow_err(msg: &str) -> crate::error::SkadooshError {
    crate::error::SkadooshError::Other(anyhow::anyhow!("calculator: {msg}"))
}

// ---------------------------------------------------------------------------
// datetime
// ---------------------------------------------------------------------------

fn now(format: Option<&str>) -> Result<String> {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs() as i64;
    let (y, mo, d, h, mi, s) = unix_to_cal(secs);

    match format.unwrap_or("friendly") {
        "iso" => Ok(format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")),
        "date" => Ok(format!("{y:04}-{mo:02}-{d:02}")),
        "time" => Ok(format!("{h:02}:{mi:02}:{s:02}")),
        "unix" => Ok(secs.to_string()),
        _ => {
            let weekday = weekday_name(y, mo, d);
            let month = month_name(mo);
            Ok(format!(
                "{weekday}, {month} {d}, {y:04} — {h:02}:{mi:02}:{s:02} UTC"
            ))
        }
    }
}

/// Howard Hinnant's `civil_from_days` algorithm — converts a Unix timestamp
/// to (year, month 1-12, day 1-31, hour, minute, second) without any
/// calendar library.
fn unix_to_cal(ts: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days_since_epoch = ts / 86400;
    let time_secs = (ts % 86400).unsigned_abs();
    let h = (time_secs / 3600) as u32;
    let mi = ((time_secs % 3600) / 60) as u32;
    let s = (time_secs % 60) as u32;

    // civil_from_days
    let z = days_since_epoch + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = (z - era * 146_097) as u32; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    (y as i32, m, d, h, mi, s)
}

fn month_name(m: u32) -> &'static str {
    match m {
        1 => "January",
        2 => "February",
        3 => "March",
        4 => "April",
        5 => "May",
        6 => "June",
        7 => "July",
        8 => "August",
        9 => "September",
        10 => "October",
        11 => "November",
        12 => "December",
        _ => "?",
    }
}

fn weekday_name(y: i32, m: u32, d: u32) -> &'static str {
    // Tomohiko Sakamoto's algorithm
    let t = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    let w = (y + y / 4 - y / 100 + y / 400 + t[(m as usize) - 1] + d as i32) % 7;
    match w.rem_euclid(7) {
        0 => "Sunday",
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6 => "Saturday",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// remember / recall
// ---------------------------------------------------------------------------

fn rem(key: &str, value: &str, store: &Mutex<MemoryStore>) -> Result<String> {
    if key.is_empty() {
        return Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
            "remember: key must not be empty"
        )));
    }
    store
        .lock()
        .map_err(|e| {
            crate::error::SkadooshError::Other(anyhow::anyhow!("memory lock poisoned: {e}"))
        })?
        .remember(key, value);
    Ok(format!("remembered '{key}'"))
}

fn rec(query: &str, store: &Mutex<MemoryStore>) -> Result<String> {
    let store = store.lock().map_err(|e| {
        crate::error::SkadooshError::Other(anyhow::anyhow!("memory lock poisoned: {e}"))
    })?;
    if store.preference_count() == 0 {
        return Ok("no memories saved yet".to_string());
    }
    let matches: Vec<String> = store
        .search_preferences(query)
        .iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect();
    if matches.is_empty() {
        Ok(format!("nothing found for '{query}'"))
    } else {
        Ok(matches.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- calculator --------------------------------------------------------

    #[test]
    fn calc_basic_ops() {
        assert_eq!(calc("2+3").unwrap(), "5");
        assert_eq!(calc("10-7").unwrap(), "3");
        assert_eq!(calc("4*5").unwrap(), "20");
        assert_eq!(calc("8/2").unwrap(), "4");
        assert_eq!(calc("7%3").unwrap(), "1");
    }

    #[test]
    fn calc_precedence() {
        assert_eq!(calc("2+3*4").unwrap(), "14");
        assert_eq!(calc("(2+3)*4").unwrap(), "20");
        assert_eq!(calc("2^3").unwrap(), "8");
        assert_eq!(calc("2^3+1").unwrap(), "9");
    }

    #[test]
    fn calc_functions() {
        assert_eq!(calc("sqrt(144)").unwrap(), "12");
        assert_eq!(calc("abs(-5)").unwrap(), "5");
        assert_eq!(calc("round(3.7)").unwrap(), "4");
        assert_eq!(calc("floor(3.9)").unwrap(), "3");
        assert_eq!(calc("ceil(3.1)").unwrap(), "4");
        assert_eq!(calc("pow(2, 8)").unwrap(), "256");
    }

    #[test]
    fn calc_unary_minus() {
        assert_eq!(calc("-5").unwrap(), "-5");
        assert_eq!(calc("-5 + 32").unwrap(), "27");
        assert_eq!(calc("3*-2").unwrap(), "-6");
        assert_eq!(calc("(-3)").unwrap(), "-3");
        assert_eq!(calc("-(2+3)").unwrap(), "-5");
        assert_eq!(calc("2+(-3)").unwrap(), "-1");
    }

    #[test]
    fn calc_constants() {
        assert!(calc("pi").unwrap().starts_with("3.14"));
    }

    #[test]
    fn calc_errors() {
        assert!(calc("1/0").is_err());
        assert!(calc("bogus").is_err());
        assert!(calc("").is_err());
        assert!(calc("(1+2").is_err());
    }

    // -- datetime ----------------------------------------------------------

    #[test]
    fn datetime_formats() {
        let iso = now(Some("iso")).unwrap();
        assert!(iso.contains('T'));
        assert!(iso.ends_with('Z'));

        let date = now(Some("date")).unwrap();
        assert_eq!(date.len(), 10);
        assert!(date.chars().nth(4) == Some('-'));

        let time = now(Some("time")).unwrap();
        assert_eq!(time.len(), 8);

        let friendly = now(Some("friendly")).unwrap();
        assert!(friendly.contains(','));
    }

    // -- remember / recall --------------------------------------------------

    #[test]
    fn remember_and_recall() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mem.json");
        let store = Mutex::new(MemoryStore::open(path));

        rem("color", "blue", &store).unwrap();
        let result = rec("color", &store).unwrap();
        assert!(result.contains("blue"));

        let empty = rec("nonexistent", &store).unwrap();
        assert!(empty.contains("nothing found"));
    }
}
