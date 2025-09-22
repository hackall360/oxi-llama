use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::rc::Rc;

use indexmap::IndexMap;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("JSON schema conversion failed:\n{0}")]
    Conversion(String),
}

#[derive(Default, Clone, Copy)]
pub struct CommonGrammarOptions {
    pub dotall: bool,
}

pub struct CommonGrammarBuilder<'a> {
    add_rule: Box<dyn Fn(&str, &str) -> String + 'a>,
    add_schema: Box<dyn Fn(&str, &Value) -> String + 'a>,
    resolve_refs: Box<dyn Fn(&mut Value) + 'a>,
}

impl<'a> CommonGrammarBuilder<'a> {
    pub fn add_rule(&self, name: &str, rule: &str) -> String {
        (self.add_rule)(name, rule)
    }

    pub fn add_schema(&self, name: &str, schema: &Value) -> String {
        (self.add_schema)(name, schema)
    }

    pub fn resolve_refs(&self, schema: &mut Value) {
        (self.resolve_refs)(schema)
    }
}

#[derive(Clone)]
struct BuiltinRule {
    content: &'static str,
    deps: &'static [&'static str],
}

fn string_repeat(s: &str, n: usize) -> String {
    s.repeat(n)
}

fn string_join(values: &[String], sep: &str) -> String {
    values.join(sep)
}

fn string_split(input: &str, sep: &str) -> Vec<String> {
    if sep.is_empty() {
        return input.chars().map(|c| c.to_string()).collect();
    }
    input.split(sep).map(|s| s.to_string()).collect()
}

static SPACE_RULE: &str = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

static PRIMITIVE_RULES: Lazy<HashMap<&'static str, BuiltinRule>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "boolean",
        BuiltinRule {
            content: "(\"true\" | \"false\") space",
            deps: &[],
        },
    );
    map.insert(
        "decimal-part",
        BuiltinRule {
            content: "[0-9]{1,16}",
            deps: &[],
        },
    );
    map.insert(
        "integral-part",
        BuiltinRule {
            content: "[0] | [1-9] [0-9]{0,15}",
            deps: &[],
        },
    );
    map.insert(
        "number",
        BuiltinRule {
            content:
                "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)? space",
            deps: &["integral-part", "decimal-part"],
        },
    );
    map.insert(
        "integer",
        BuiltinRule {
            content: "(\"-\"? integral-part) space",
            deps: &["integral-part"],
        },
    );
    map.insert(
        "value",
        BuiltinRule {
            content: "object | array | string | number | boolean | null",
            deps: &["object", "array", "string", "number", "boolean", "null"],
        },
    );
    map.insert(
        "object",
        BuiltinRule {
            content: "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space",
            deps: &["string", "value"],
        },
    );
    map.insert(
        "array",
        BuiltinRule {
            content: "\"[\" space ( value (\",\" space value)* )? \"]\" space",
            deps: &["value"],
        },
    );
    map.insert(
        "uuid",
        BuiltinRule {
            content: "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\" space",
            deps: &[],
        },
    );
    map.insert(
        "char",
        BuiltinRule {
            content: "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})",
            deps: &[],
        },
    );
    map.insert(
        "string",
        BuiltinRule {
            content: "\"\\\"\" char* \"\\\"\" space",
            deps: &["char"],
        },
    );
    map.insert(
        "null",
        BuiltinRule {
            content: "\"null\" space",
            deps: &[],
        },
    );
    map
});

static STRING_FORMAT_RULES: Lazy<HashMap<&'static str, BuiltinRule>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "date",
        BuiltinRule {
            content: "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )",
            deps: &[],
        },
    );
    map.insert(
        "time",
        BuiltinRule {
            content: "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] )",
            deps: &[],
        },
    );
    map.insert(
        "date-time",
        BuiltinRule {
            content: "date \"T\" time",
            deps: &["date", "time"],
        },
    );
    map.insert(
        "date-string",
        BuiltinRule {
            content: "\"\\\"\" date \"\\\"\" space",
            deps: &["date"],
        },
    );
    map.insert(
        "time-string",
        BuiltinRule {
            content: "\"\\\"\" time \"\\\"\" space",
            deps: &["time"],
        },
    );
    map.insert(
        "date-time-string",
        BuiltinRule {
            content: "\"\\\"\" date-time \"\\\"\" space",
            deps: &["date-time"],
        },
    );
    map
});

static RESERVED_NAMES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    let mut set = HashSet::new();
    set.insert("root");
    for key in PRIMITIVE_RULES.keys() {
        set.insert(*key);
    }
    for key in STRING_FORMAT_RULES.keys() {
        set.insert(*key);
    }
    set
});

static INVALID_RULE_CHARS_RE: Lazy<Regex> = Lazy::new(|| Regex::new("[^a-zA-Z0-9-]+").unwrap());
static GRAMMAR_LITERAL_ESCAPE_RE: Lazy<Regex> = Lazy::new(|| Regex::new("[\\r\\n\"]").unwrap());
static UUID_FORMAT_RE: Lazy<Regex> = Lazy::new(|| Regex::new("^uuid[1-5]?$").unwrap());
static NON_LITERAL_SET: Lazy<HashSet<char>> = Lazy::new(|| {
    ['|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?']
        .into_iter()
        .collect()
});
static ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS: Lazy<HashSet<char>> = Lazy::new(|| {
    [
        '^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?',
    ]
    .into_iter()
    .collect()
});

fn grammar_literal_escape(c: char) -> &'static str {
    match c {
        '\r' => "\\r",
        '\n' => "\\n",
        '"' => "\\\"",
        '-' => "\\-",
        ']' => "\\]",
        _ => panic!("unexpected literal escape: {c}"),
    }
}

fn replace_pattern(
    input: &str,
    regex: &Regex,
    replacement: impl Fn(&regex::Captures) -> String,
) -> String {
    let mut result = String::with_capacity(input.len());
    let mut last = 0;
    for cap in regex.captures_iter(input) {
        let m = cap.get(0).unwrap();
        result.push_str(&input[last..m.start()]);
        result.push_str(&replacement(&cap));
        last = m.end();
    }
    result.push_str(&input[last..]);
    result
}

fn format_literal(literal: &str) -> String {
    let escaped = replace_pattern(literal, &GRAMMAR_LITERAL_ESCAPE_RE, |caps| {
        let c = caps.get(0).unwrap().as_str().chars().next().unwrap();
        grammar_literal_escape(c).to_string()
    });
    format!("\"{}\"", escaped)
}

fn build_repetition(
    item_rule: &str,
    min_items: i32,
    max_items: i32,
    separator_rule: &str,
) -> String {
    let has_max = max_items != i32::MAX;

    if max_items == 0 {
        return String::new();
    }
    if min_items == 0 && max_items == 1 {
        return format!("{}?", item_rule);
    }

    if separator_rule.is_empty() {
        if min_items == 1 && !has_max {
            format!("{}+", item_rule)
        } else if min_items == 0 && !has_max {
            format!("{}*", item_rule)
        } else {
            let mut s = format!("{}{{{}", item_rule, min_items);
            s.push(',');
            if has_max {
                s.push_str(&max_items.to_string());
            }
            s.push('}');
            s
        }
    } else {
        let tail = build_repetition(
            &format!("({} {})", separator_rule, item_rule),
            if min_items == 0 { 0 } else { min_items - 1 },
            if has_max { max_items - 1 } else { max_items },
            "",
        );
        let combined = if tail.is_empty() {
            item_rule.to_string()
        } else {
            format!("{} {}", item_rule, tail)
        };
        if min_items == 0 {
            format!("({})?", combined)
        } else {
            combined
        }
    }
}

fn digit_range(out: &mut String, from: char, to: char) {
    out.push('[');
    if from == to {
        out.push(from);
    } else {
        out.push(from);
        out.push('-');
        out.push(to);
    }
    out.push(']');
}

fn more_digits(out: &mut String, min_digits: i32, max_digits: i32) {
    out.push_str("[0-9]");
    if min_digits == max_digits && min_digits == 1 {
        return;
    }
    out.push('{');
    write!(out, "{}", min_digits).unwrap();
    if max_digits != min_digits {
        out.push(',');
        if max_digits != i32::MAX {
            write!(out, "{}", max_digits).unwrap();
        }
    }
    out.push('}');
}

fn uniform_range(out: &mut String, from: &str, to: &str) {
    let mut i = 0usize;
    let from_bytes = from.as_bytes();
    let to_bytes = to.as_bytes();
    while i < from.len() && i < to.len() && from_bytes[i] == to_bytes[i] {
        i += 1;
    }
    if i > 0 {
        out.push('"');
        out.push_str(&from[..i]);
        out.push('"');
    }
    if i < from.len() && i < to.len() {
        if i > 0 {
            out.push(' ');
        }
        let sub_len = from.len() - i - 1;
        if sub_len > 0 {
            let from_sub = &from[i + 1..];
            let to_sub = &to[i + 1..];
            let sub_zeros = string_repeat("0", sub_len);
            let sub_nines = string_repeat("9", sub_len);
            let mut to_reached = false;
            out.push('(');
            if from_sub == sub_zeros {
                digit_range(out, char::from(from_bytes[i]), char::from(to_bytes[i] - 1));
                out.push(' ');
                more_digits(out, sub_len as i32, sub_len as i32);
            } else {
                digit_range(out, char::from(from_bytes[i]), char::from(from_bytes[i]));
                out.push(' ');
                out.push('(');
                uniform_range(out, from_sub, &sub_nines);
                out.push(')');
                if from_bytes[i] < to_bytes[i] - 1 {
                    out.push_str(" | ");
                    if to_sub == sub_nines {
                        digit_range(out, char::from(from_bytes[i] + 1), char::from(to_bytes[i]));
                        to_reached = true;
                    } else {
                        digit_range(
                            out,
                            char::from(from_bytes[i] + 1),
                            char::from(to_bytes[i] - 1),
                        );
                    }
                    out.push(' ');
                    more_digits(out, sub_len as i32, sub_len as i32);
                }
            }
            if !to_reached {
                out.push_str(" | ");
                digit_range(out, char::from(to_bytes[i]), char::from(to_bytes[i]));
                out.push(' ');
                uniform_range(out, sub_zeros.as_str(), to_sub);
            }
            out.push(')');
        } else {
            digit_range(out, char::from(from_bytes[i]), char::from(to_bytes[i]));
        }
    }
}

fn build_min_max_int(
    min_value: i32,
    max_value: i32,
    out: &mut String,
    decimals_left: i32,
    top_level: bool,
) {
    let has_min = min_value != i32::MIN;
    let has_max = max_value != i32::MAX;

    if has_min && has_max {
        if min_value < 0 && max_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(-max_value, -min_value, out, decimals_left, true);
            out.push(')');
            return;
        }
        let mut min_value = min_value;
        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(0, -min_value, out, decimals_left, true);
            out.push_str(") | ");
            min_value = 0;
        }
        let mut min_s = min_value.to_string();
        let max_s = max_value.to_string();
        let min_digits = min_s.len();
        let max_digits = max_s.len();
        for digits in min_digits..max_digits {
            uniform_range(out, &min_s, &string_repeat("9", digits));
            min_s = format!("1{}", string_repeat("0", digits));
            out.push_str(" | ");
        }
        uniform_range(out, &min_s, &max_s);
        return;
    }

    let less_decimals = (decimals_left - 1).max(1);
    if has_min {
        if min_value < 0 {
            out.push_str("\"-\" (");
            build_min_max_int(i32::MIN, -min_value, out, decimals_left, false);
            out.push_str(") | [0] | [1-9] ");
            more_digits(out, 0, decimals_left - 1);
        } else if min_value == 0 {
            if top_level {
                out.push_str("[0] | [1-9] ");
                more_digits(out, 0, less_decimals);
            } else {
                more_digits(out, 1, decimals_left);
            }
        } else if min_value <= 9 {
            let c = char::from(b'0' + min_value as u8);
            let range_start = if top_level { '1' } else { '0' };
            if c > range_start {
                digit_range(out, range_start, char::from((c as u8) - 1));
                out.push(' ');
                more_digits(out, 1, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, '9');
            out.push(' ');
            more_digits(out, 0, less_decimals);
        } else {
            let min_s = min_value.to_string();
            let len = min_s.len();
            let c = min_s.as_bytes()[0] as char;
            if c > '1' {
                let start = if top_level { '1' } else { '0' };
                digit_range(out, start, char::from((c as u8) - 1));
                out.push(' ');
                more_digits(out, len as i32, less_decimals);
                out.push_str(" | ");
            }
            digit_range(out, c, c);
            out.push_str(" (");
            let rest = min_s[1..].parse::<i32>().unwrap();
            build_min_max_int(rest, i32::MAX, out, less_decimals, false);
            out.push(')');
            if c < '9' {
                out.push_str(" | ");
                digit_range(out, char::from((c as u8) + 1), '9');
                out.push(' ');
                more_digits(out, (len - 1) as i32, less_decimals);
            }
        }
        return;
    }

    if has_max {
        if max_value >= 0 {
            if top_level {
                out.push_str("\"-\" [1-9] ");
                more_digits(out, 0, less_decimals);
                out.push_str(" | ");
            }
            build_min_max_int(0, max_value, out, decimals_left, true);
        } else {
            out.push_str("\"-\" (");
            build_min_max_int(-max_value, i32::MAX, out, decimals_left, false);
            out.push(')');
        }
        return;
    }

    panic!("At least one of min_value or max_value must be set");
}

fn is_reserved_name(name: &str) -> bool {
    RESERVED_NAMES.contains(name)
}

struct SchemaConverter<'a> {
    fetch_json: Box<dyn Fn(&str) -> Value + 'a>,
    dotall: bool,
    rules: IndexMap<String, String>,
    refs: HashMap<String, Value>,
    refs_being_resolved: HashSet<String>,
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl<'a> SchemaConverter<'a> {
    fn new(fetch_json: impl Fn(&str) -> Value + 'a, dotall: bool) -> Self {
        let mut rules = IndexMap::new();
        rules.insert("space".to_string(), SPACE_RULE.to_string());
        Self {
            fetch_json: Box::new(fetch_json),
            dotall,
            rules,
            refs: HashMap::new(),
            refs_being_resolved: HashSet::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn add_rule(&mut self, name: &str, rule: &str) -> String {
        let esc_name = INVALID_RULE_CHARS_RE.replace_all(name, "-").to_string();
        let rule_owned = rule.to_string();
        if !self.rules.contains_key(&esc_name) || self.rules.get(&esc_name) == Some(&rule_owned) {
            self.rules.insert(esc_name.clone(), rule_owned);
            esc_name
        } else {
            let base = esc_name.clone();
            let mut i = 0;
            loop {
                let candidate = format!("{}{}", base, i);
                match self.rules.get(&candidate) {
                    None => {
                        self.rules.insert(candidate.clone(), rule_owned.clone());
                        break candidate;
                    }
                    Some(existing) if *existing == rule_owned => break candidate,
                    _ => i += 1,
                }
            }
        }
    }

    fn add_primitive(&mut self, name: &str, rule: &BuiltinRule) -> String {
        let n = self.add_rule(name, rule.content);
        for dep in rule.deps {
            if self.rules.contains_key(*dep) {
                continue;
            }
            if let Some(dep_rule) = PRIMITIVE_RULES
                .get(*dep)
                .or_else(|| STRING_FORMAT_RULES.get(*dep))
            {
                self.add_primitive(dep, dep_rule);
            } else {
                self.errors.push(format!("Rule {} not known", dep));
            }
        }
        n
    }

    fn generate_union_rule(&mut self, name: &str, alt_schemas: &[Value]) -> String {
        let mut rules = Vec::with_capacity(alt_schemas.len());
        for (i, schema) in alt_schemas.iter().enumerate() {
            let alt_name = format!(
                "{}{}{}",
                name,
                if name.is_empty() { "alternative-" } else { "-" },
                i
            );
            rules.push(self.visit(schema, &alt_name));
        }
        string_join(&rules, " | ")
    }

    fn visit_pattern(&mut self, pattern: &str, name: &str) -> String {
        use std::collections::hash_map::Entry;

        if pattern.len() < 2 || !pattern.starts_with('^') || !pattern.ends_with('$') {
            self.errors
                .push("Pattern must start with '^' and end with '$'".to_string());
            return String::new();
        }

        let sub_pattern = &pattern[1..pattern.len() - 1];

        #[derive(Clone)]
        struct LiteralOrRule {
            content: String,
            is_literal: bool,
        }

        struct PatternParser<'a, 'b> {
            converter: &'b mut SchemaConverter<'a>,
            pattern: Vec<u8>,
            length: usize,
            index: usize,
            sub_rule_ids: HashMap<String, String>,
            next_sub_rule_id: usize,
        }

        impl<'a, 'b> PatternParser<'a, 'b> {
            fn new(converter: &'b mut SchemaConverter<'a>, pattern: &str) -> Self {
                let bytes = pattern.as_bytes().to_vec();
                Self {
                    converter,
                    length: bytes.len(),
                    pattern: bytes,
                    index: 0,
                    sub_rule_ids: HashMap::new(),
                    next_sub_rule_id: 0,
                }
            }

            fn to_rule(&self, item: &LiteralOrRule) -> String {
                if item.is_literal {
                    format!("\"{}\"", item.content)
                } else {
                    item.content.clone()
                }
            }

            fn dot_rule(&mut self) -> String {
                let rule = if self.converter.dotall {
                    "[\\U00000000-\\U0010FFFF]"
                } else {
                    "[^\\x0A\\x0D]"
                };
                self.converter.add_rule("dot", rule)
            }

            fn join_seq(&mut self, seq: Vec<LiteralOrRule>) -> LiteralOrRule {
                let mut ret = Vec::new();
                let mut literal = String::new();
                let flush_literal = |literal: &mut String, ret: &mut Vec<LiteralOrRule>| {
                    if literal.is_empty() {
                        return;
                    }
                    ret.push(LiteralOrRule {
                        content: literal.clone(),
                        is_literal: true,
                    });
                    literal.clear();
                };

                for item in seq {
                    if item.is_literal {
                        literal.push_str(&item.content);
                    } else {
                        flush_literal(&mut literal, &mut ret);
                        ret.push(item);
                    }
                }
                flush_literal(&mut literal, &mut ret);

                let parts: Vec<String> = ret.iter().map(|item| self.to_rule(item)).collect();
                LiteralOrRule {
                    content: string_join(&parts, " "),
                    is_literal: false,
                }
            }

            fn parse_sequence(&mut self, name: &str, start: usize) -> LiteralOrRule {
                let mut seq: Vec<LiteralOrRule> = Vec::new();
                while self.index < self.length {
                    let c = self.pattern[self.index] as char;
                    match c {
                        '.' => {
                            let dot = self.dot_rule();
                            seq.push(LiteralOrRule {
                                content: dot,
                                is_literal: false,
                            });
                            self.index += 1;
                        }
                        '(' => {
                            self.index += 1;
                            if self.index < self.length && self.pattern[self.index] as char == '?' {
                                self.converter
                                    .warnings
                                    .push("Unsupported pattern syntax".to_string());
                            }
                            let inner_start = self.index;
                            let inner = self.parse_sequence(name, inner_start);
                            let inner_rule = format!("({})", self.to_rule(&inner));
                            seq.push(LiteralOrRule {
                                content: inner_rule,
                                is_literal: false,
                            });
                        }
                        ')' => {
                            self.index += 1;
                            if start > 0 && (self.pattern[start - 1] as char) != '(' {
                                self.converter
                                    .errors
                                    .push("Unbalanced parentheses".to_string());
                            }
                            return self.join_seq(seq);
                        }
                        '[' => {
                            let mut square = String::from("[");
                            self.index += 1;
                            while self.index < self.length
                                && self.pattern[self.index] as char != ']'
                            {
                                if self.pattern[self.index] as char == '\\'
                                    && self.index + 1 < self.length
                                {
                                    square.push('\\');
                                    self.index += 1;
                                    square.push(self.pattern[self.index] as char);
                                    self.index += 1;
                                } else {
                                    square.push(self.pattern[self.index] as char);
                                    self.index += 1;
                                }
                            }
                            if self.index >= self.length {
                                self.converter
                                    .errors
                                    .push("Unbalanced square brackets".to_string());
                            }
                            square.push(']');
                            if self.index < self.length {
                                self.index += 1;
                            }
                            seq.push(LiteralOrRule {
                                content: square,
                                is_literal: false,
                            });
                        }
                        '|' => {
                            seq.push(LiteralOrRule {
                                content: "|".to_string(),
                                is_literal: false,
                            });
                            self.index += 1;
                        }
                        '*' | '+' | '?' => {
                            if let Some(last) = seq.pop() {
                                let updated = LiteralOrRule {
                                    content: format!("{}{}", self.to_rule(&last), c),
                                    is_literal: false,
                                };
                                seq.push(updated);
                            }
                            self.index += 1;
                        }
                        '{' => {
                            let mut curly = String::from("{");
                            self.index += 1;
                            while self.index < self.length
                                && self.pattern[self.index] as char != '}'
                            {
                                curly.push(self.pattern[self.index] as char);
                                self.index += 1;
                            }
                            if self.index >= self.length {
                                self.converter
                                    .errors
                                    .push("Unbalanced curly brackets".to_string());
                            }
                            curly.push('}');
                            if self.index < self.length {
                                self.index += 1;
                            }

                            let nums = string_split(&curly[1..curly.len() - 1], ",");
                            let mut min_times = 0i32;
                            let mut max_times = i32::MAX;
                            let parse_component = |s: &str| -> Result<Option<i32>, ()> {
                                let trimmed = s.trim();
                                if trimmed.is_empty() {
                                    Ok(None)
                                } else {
                                    trimmed.parse::<i32>().map(Some).map_err(|_| ())
                                }
                            };
                            let mut error_message: Option<&'static str> = None;
                            if nums.len() == 1 {
                                match parse_component(&nums[0]) {
                                    Ok(Some(v)) => {
                                        min_times = v;
                                        max_times = v;
                                    }
                                    Ok(None) => {
                                        min_times = 0;
                                        max_times = 0;
                                    }
                                    Err(_) => {
                                        error_message = Some("Invalid number in curly brackets")
                                    }
                                }
                            } else if nums.len() == 2 {
                                match parse_component(&nums[0]) {
                                    Ok(Some(v)) => min_times = v,
                                    Ok(None) => {}
                                    Err(_) => {
                                        error_message = Some("Invalid number in curly brackets")
                                    }
                                }
                                if error_message.is_none() {
                                    match parse_component(&nums[1]) {
                                        Ok(Some(v)) => max_times = v,
                                        Ok(None) => {}
                                        Err(_) => {
                                            error_message = Some("Invalid number in curly brackets")
                                        }
                                    }
                                }
                            } else {
                                error_message = Some("Wrong number of values in curly brackets");
                            }

                            if let Some(msg) = error_message {
                                self.converter.errors.push(msg.to_string());
                                return LiteralOrRule {
                                    content: String::new(),
                                    is_literal: false,
                                };
                            }

                            if let Some(last) = seq.last_mut() {
                                let sub = last.content.clone();
                                let sub_is_literal = last.is_literal;
                                let mut replacement = sub.clone();
                                if !sub_is_literal {
                                    let entry = self.sub_rule_ids.entry(sub.clone());
                                    let rule_name = match entry {
                                        Entry::Occupied(o) => o.get().clone(),
                                        Entry::Vacant(v) => {
                                            let sub_name =
                                                format!("{}-{}", name, self.next_sub_rule_id);
                                            self.next_sub_rule_id += 1;
                                            let id = self.converter.add_rule(&sub_name, &sub);
                                            v.insert(id.clone());
                                            id
                                        }
                                    };
                                    replacement = rule_name;
                                }
                                let base = if sub_is_literal {
                                    format!("\"{}\"", replacement)
                                } else {
                                    replacement
                                };
                                last.content = build_repetition(&base, min_times, max_times, "");
                                last.is_literal = false;
                            }
                        }
                        _ => {
                            let mut literal = String::new();
                            while self.index < self.length {
                                if self.pattern[self.index] as char == '\\'
                                    && self.index + 1 < self.length
                                {
                                    let next = self.pattern[self.index + 1] as char;
                                    if ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.contains(&next) {
                                        literal.push(next);
                                        self.index += 2;
                                    } else {
                                        literal.push('\\');
                                        literal.push(next);
                                        self.index += 2;
                                    }
                                } else if self.pattern[self.index] as char == '"' {
                                    literal.push_str("\\\"");
                                    self.index += 1;
                                } else {
                                    let ch = self.pattern[self.index] as char;
                                    let next_special = if self.index + 1 < self.length {
                                        let next = self.pattern[self.index + 1] as char;
                                        NON_LITERAL_SET.contains(&next)
                                    } else {
                                        false
                                    };
                                    if !NON_LITERAL_SET.contains(&ch)
                                        && (self.index + 1 == self.length
                                            || literal.is_empty()
                                            || self.pattern[self.index + 1] as char == '.'
                                            || !next_special)
                                    {
                                        literal.push(ch);
                                        self.index += 1;
                                    } else {
                                        break;
                                    }
                                }
                            }
                            if !literal.is_empty() {
                                seq.push(LiteralOrRule {
                                    content: literal,
                                    is_literal: true,
                                });
                            }
                        }
                    }
                }

                if start > 0 {
                    self.converter
                        .errors
                        .push("Unbalanced parentheses".to_string());
                }
                self.join_seq(seq)
            }
        }

        let rule_content = {
            let mut parser = PatternParser::new(self, sub_pattern);
            let parsed = parser.parse_sequence(name, 0);
            let body = parser.to_rule(&parsed);
            format!("\"\\\"\" ({}) \"\\\"\" space", body)
        };

        self.add_rule(name, &rule_content)
    }

    fn not_strings(&mut self, strings: &[String]) -> String {
        #[derive(Default)]
        struct TrieNode {
            children: std::collections::BTreeMap<char, TrieNode>,
            is_end_of_string: bool,
        }

        impl TrieNode {
            fn insert(&mut self, s: &str) {
                let mut node = self;
                for ch in s.chars() {
                    node = node.children.entry(ch).or_insert_with(TrieNode::default);
                }
                node.is_end_of_string = true;
            }
        }

        let mut trie = TrieNode::default();
        for s in strings {
            trie.insert(s);
        }

        let char_rule = self.add_primitive("char", PRIMITIVE_RULES.get("char").unwrap());
        let mut out = String::new();
        out.push_str("[\"] ( ");

        fn visit(node: &TrieNode, out: &mut String, char_rule: &str) {
            let mut rejects = String::new();
            let mut first = true;
            for (ch, child) in &node.children {
                rejects.push(*ch);
                if !first {
                    out.push_str(" | ");
                }
                first = false;
                write!(out, "[{}]", ch).unwrap();
                if !child.children.is_empty() {
                    out.push_str(" (");
                    visit(child, out, char_rule);
                    out.push(')');
                } else if child.is_end_of_string {
                    write!(out, " {}+", char_rule).unwrap();
                }
            }
            if !node.children.is_empty() {
                if !first {
                    out.push_str(" | ");
                }
                write!(out, "[^\"{}] {}*", rejects, char_rule).unwrap();
            }
        }

        visit(&trie, &mut out, &char_rule);

        out.push_str(" )");
        if !trie.is_end_of_string {
            out.push('?');
        }
        out.push_str(" [\"] space");
        out
    }

    fn resolve_ref(&mut self, reference: &str) -> String {
        let mut ref_name = reference
            .rsplit('/')
            .next()
            .unwrap_or(reference)
            .to_string();
        if !self.rules.contains_key(&ref_name) && !self.refs_being_resolved.contains(reference) {
            self.refs_being_resolved.insert(reference.to_string());
            if let Some(resolved) = self.refs.get(reference).cloned() {
                ref_name = self.visit(&resolved, &ref_name);
            }
            self.refs_being_resolved.remove(reference);
        }
        ref_name
    }

    fn build_object_rule(
        &mut self,
        properties: &[(String, Value)],
        required: &HashSet<String>,
        name: &str,
        additional_properties: Option<&Value>,
    ) -> String {
        let mut required_props = Vec::new();
        let mut optional_props = Vec::new();
        let mut prop_kv_rule_names: HashMap<String, String> = HashMap::new();
        let mut prop_names = Vec::new();

        for (prop_name, prop_schema) in properties {
            let sub_name = format!(
                "{}{}{}",
                name,
                if name.is_empty() { "" } else { "-" },
                prop_name
            );
            let prop_rule_name = self.visit(prop_schema, &sub_name);
            let kv_rule = self.add_rule(
                &format!(
                    "{}{}{}-kv",
                    name,
                    if name.is_empty() { "" } else { "-" },
                    prop_name
                ),
                &format!(
                    "{} space \":\" space {}",
                    format_literal(&serde_json::json!(prop_name).to_string()),
                    prop_rule_name
                ),
            );
            if required.contains(prop_name) {
                required_props.push(prop_name.clone());
            } else {
                optional_props.push(prop_name.clone());
            }
            prop_kv_rule_names.insert(prop_name.clone(), kv_rule);
            prop_names.push(prop_name.clone());
        }

        if let Some(additional) = additional_properties {
            let sub_name = format!(
                "{}{}additional",
                name,
                if name.is_empty() { "" } else { "-" }
            );
            let value_rule = if additional.is_object() {
                self.visit(additional, &format!("{}-value", sub_name))
            } else {
                self.add_primitive("value", PRIMITIVE_RULES.get("value").unwrap())
            };
            let key_rule = if prop_names.is_empty() {
                self.add_primitive("string", PRIMITIVE_RULES.get("string").unwrap())
            } else {
                let rule = self.not_strings(&prop_names);
                self.add_rule(&format!("{}-k", sub_name), &rule)
            };
            let kv_rule = self.add_rule(
                &format!("{}-kv", sub_name),
                &format!("{} \":\" space {}", key_rule, value_rule),
            );
            prop_kv_rule_names.insert("*".to_string(), kv_rule);
            optional_props.push("*".to_string());
        }

        let mut rule = String::from("\"{\" space ");
        for (i, prop) in required_props.iter().enumerate() {
            if i > 0 {
                rule.push_str(" \",\" space ");
            }
            rule.push_str(prop_kv_rule_names.get(prop).unwrap());
        }

        if !optional_props.is_empty() {
            rule.push_str(" (");
            if !required_props.is_empty() {
                rule.push_str(" \",\" space ( ");
            }

            fn recursive_refs(
                converter: &mut SchemaConverter,
                optional_props: &[String],
                prop_kv_rule_names: &HashMap<String, String>,
                name: &str,
                index: usize,
                first_is_optional: bool,
            ) -> String {
                if index >= optional_props.len() {
                    return String::new();
                }
                let k = &optional_props[index];
                let kv_rule_name = prop_kv_rule_names.get(k).unwrap();
                let comma_ref = format!("( \",\" space {} )", kv_rule_name);
                let mut res = if first_is_optional {
                    if k == "*" {
                        format!("{}*", comma_ref)
                    } else {
                        format!("{}?", comma_ref)
                    }
                } else if k == "*" {
                    format!("{} {}*", kv_rule_name, comma_ref)
                } else {
                    kv_rule_name.clone()
                };
                if index + 1 < optional_props.len() {
                    let rest = recursive_refs(
                        converter,
                        optional_props,
                        prop_kv_rule_names,
                        name,
                        index + 1,
                        true,
                    );
                    if !rest.is_empty() {
                        let rest_name = format!(
                            "{}{}{}-rest",
                            name,
                            if name.is_empty() { "" } else { "-" },
                            k
                        );
                        let rest_rule = converter.add_rule(&rest_name, &rest);
                        if !res.is_empty() {
                            res.push(' ');
                        }
                        res.push_str(&rest_rule);
                    }
                }
                res
            }

            for i in 0..optional_props.len() {
                if i > 0 {
                    rule.push_str(" | ");
                }
                let res =
                    recursive_refs(self, &optional_props, &prop_kv_rule_names, name, i, false);
                rule.push_str(&res);
            }
            if !required_props.is_empty() {
                rule.push_str(" )");
            }
            rule.push_str(" )?");
        }

        rule.push_str(" \"}\" space");
        rule
    }

    fn resolve_refs(&mut self, schema: &mut Value, url: &str) {
        fn visit(converter: &mut SchemaConverter, node: &mut Value, root: &Value, url: &str) {
            match node {
                Value::Array(items) => {
                    for item in items.iter_mut() {
                        visit(converter, item, root, url);
                    }
                }
                Value::Object(map) => {
                    if let Some(Value::String(ref_value)) = map.get("$ref").cloned() {
                        let mut ref_url = ref_value;
                        if converter.refs.contains_key(&ref_url) {
                            return;
                        }

                        let (target, pointer) = if ref_url.starts_with("https://") {
                            let base_url = ref_url.split('#').next().unwrap_or("").to_string();
                            let target =
                                if let Some(existing) = converter.refs.get(&base_url).cloned() {
                                    existing
                                } else {
                                    let mut referenced = (converter.fetch_json)(&ref_url);
                                    converter.resolve_refs(&mut referenced, &base_url);
                                    converter.refs.insert(base_url.clone(), referenced.clone());
                                    referenced
                                };
                            let pointer = ref_url.split('#').nth(1).unwrap_or("").to_string();
                            if pointer.is_empty() {
                                return;
                            }
                            (target, pointer)
                        } else if ref_url.starts_with("#/") {
                            let target = root.clone();
                            let new_ref = format!("{}{}", url, ref_url);
                            map.insert("$ref".to_string(), Value::String(new_ref.clone()));
                            ref_url = new_ref;
                            let pointer = ref_url.split('#').nth(1).unwrap_or("").to_string();
                            if pointer.is_empty() {
                                return;
                            }
                            (target, pointer)
                        } else {
                            converter
                                .errors
                                .push(format!("Unsupported ref: {}", ref_url));
                            return;
                        };

                        if let Some(target_value) = target.pointer(&pointer).cloned() {
                            converter.refs.insert(ref_url.clone(), target_value);
                        } else {
                            converter.errors.push(format!(
                                "Error resolving ref {}: {} not found",
                                ref_url, pointer
                            ));
                        }
                    } else {
                        let keys: Vec<String> = map.keys().cloned().collect();
                        for key in keys {
                            if let Some(value) = map.get_mut(&key) {
                                visit(converter, value, root, url);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let root_copy = schema.clone();
        visit(self, schema, &root_copy, url);
    }

    fn generate_constant_rule(&self, value: &Value) -> String {
        format_literal(&value.to_string())
    }

    fn visit(&mut self, schema: &Value, name: &str) -> String {
        let schema_type_value = schema.get("type").cloned().unwrap_or(Value::Null);
        let schema_format = schema
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let rule_name = if is_reserved_name(name) {
            format!("{}-", name)
        } else if name.is_empty() {
            "root".to_string()
        } else {
            name.to_string()
        };

        if let Some(Value::String(reference)) = schema.get("$ref") {
            let resolved = self.resolve_ref(reference);
            return self.add_rule(&rule_name, &resolved);
        } else if let Some(Value::Array(one_of)) = schema.get("oneOf") {
            let rules = one_of.iter().cloned().collect::<Vec<_>>();
            let rule = self.generate_union_rule(name, &rules);
            return self.add_rule(&rule_name, &rule);
        } else if let Some(Value::Array(any_of)) = schema.get("anyOf") {
            let rules = any_of.iter().cloned().collect::<Vec<_>>();
            let rule = self.generate_union_rule(name, &rules);
            return self.add_rule(&rule_name, &rule);
        } else if matches!(schema_type_value, Value::Array(_)) {
            let mut schema_types = Vec::new();
            if let Value::Array(items) = schema_type_value {
                for t in items {
                    let mut schema_copy = schema.clone();
                    if let Some(obj) = schema_copy.as_object_mut() {
                        obj.insert("type".to_string(), t);
                    }
                    schema_types.push(schema_copy);
                }
            }
            let rule = self.generate_union_rule(name, &schema_types);
            return self.add_rule(&rule_name, &rule);
        } else if let Some(const_value) = schema.get("const") {
            let rule = format!("{} space", self.generate_constant_rule(const_value));
            return self.add_rule(&rule_name, &rule);
        } else if let Some(Value::Array(enum_values)) = schema.get("enum") {
            let mut values = Vec::new();
            for v in enum_values {
                values.push(self.generate_constant_rule(v));
            }
            let rule = format!("({}) space", string_join(&values, " | "));
            return self.add_rule(&rule_name, &rule);
        }

        let schema_type_str = schema_type_value.as_str();
        let schema_type_is_null = schema_type_value.is_null();

        if (schema_type_is_null || schema_type_str == Some("object"))
            && (schema.get("properties").is_some()
                || (schema.get("additionalProperties").is_some()
                    && schema.get("additionalProperties") != Some(&Value::Bool(true))))
        {
            let mut required = HashSet::new();
            if let Some(Value::Array(items)) = schema.get("required") {
                for item in items {
                    if let Some(s) = item.as_str() {
                        required.insert(s.to_string());
                    }
                }
            }
            let mut properties = Vec::new();
            if let Some(Value::Object(map)) = schema.get("properties") {
                for (key, value) in map {
                    properties.push((key.clone(), value.clone()));
                }
            }
            let additional_option = schema
                .get("additionalProperties")
                .filter(|value| !matches!(value, Value::Bool(false)));
            let rule = self.build_object_rule(&properties, &required, name, additional_option);
            return self.add_rule(&rule_name, &rule);
        } else if (schema_type_is_null || schema_type_str == Some("object"))
            && schema.get("allOf").is_some()
        {
            let mut required = HashSet::new();
            let mut properties = Vec::new();
            fn add_component(
                converter: &mut SchemaConverter,
                comp_schema: &Value,
                is_required: bool,
                required: &mut HashSet<String>,
                properties: &mut Vec<(String, Value)>,
            ) {
                if let Some(Value::String(reference)) = comp_schema.get("$ref") {
                    if let Some(resolved) = converter.refs.get(reference).cloned() {
                        add_component(converter, &resolved, is_required, required, properties);
                    }
                } else if let Some(Value::Object(map)) = comp_schema.get("properties") {
                    for (key, value) in map {
                        if is_required {
                            required.insert(key.clone());
                        }
                        properties.push((key.clone(), value.clone()));
                    }
                }
            }
            if let Some(Value::Array(items)) = schema.get("allOf") {
                for comp in items {
                    if let Some(Value::Array(any_items)) = comp.get("anyOf") {
                        for item in any_items {
                            add_component(self, item, false, &mut required, &mut properties);
                        }
                    } else {
                        add_component(self, comp, true, &mut required, &mut properties);
                    }
                }
            }
            let rule = self.build_object_rule(&properties, &required, name, None);
            return self.add_rule(&rule_name, &rule);
        } else if (schema_type_is_null || schema_type_str == Some("array"))
            && (schema.get("items").is_some() || schema.get("prefixItems").is_some())
        {
            let items = schema
                .get("items")
                .unwrap_or_else(|| schema.get("prefixItems").unwrap());
            if let Some(array) = items.as_array() {
                let mut rule = String::from("\"[\" space ");
                for (i, item) in array.iter().enumerate() {
                    if i > 0 {
                        rule.push_str(" \",\" space ");
                    }
                    let item_rule = self.visit(
                        item,
                        &format!(
                            "{}{}tuple-{}",
                            name,
                            if name.is_empty() { "" } else { "-" },
                            i
                        ),
                    );
                    rule.push_str(&item_rule);
                }
                rule.push_str(" \"]\" space");
                return self.add_rule(&rule_name, &rule);
            } else {
                let item_rule_name = self.visit(
                    items,
                    &format!("{}{}item", name, if name.is_empty() { "" } else { "-" }),
                );
                let min_items = schema.get("minItems").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let max_items = schema
                    .get("maxItems")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(i32::MAX);
                let repetition =
                    build_repetition(&item_rule_name, min_items, max_items, "\",\" space");
                let rule = format!("\"[\" space {} \"]\" space", repetition);
                return self.add_rule(&rule_name, &rule);
            }
        } else if schema_type_is_null || schema_type_str == Some("string") {
            if let Some(Value::String(pattern)) = schema.get("pattern") {
                return self.visit_pattern(pattern, &rule_name);
            }
            if UUID_FORMAT_RE.is_match(&schema_format) {
                return self.add_primitive(
                    if rule_name == "root" { "root" } else { "uuid" },
                    PRIMITIVE_RULES.get("uuid").unwrap(),
                );
            }
            let prim_name = format!("{}-string", schema_format);
            if let Some(prim_rule) = STRING_FORMAT_RULES.get(prim_name.as_str()) {
                let rule = self.add_primitive(&prim_name, prim_rule);
                return self.add_rule(&rule_name, &rule);
            }
            if schema_type_str == Some("string")
                && (schema.get("minLength").is_some() || schema.get("maxLength").is_some())
            {
                let char_rule = self.add_primitive("char", PRIMITIVE_RULES.get("char").unwrap());
                let min_len = schema
                    .get("minLength")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                let max_len = schema
                    .get("maxLength")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(i32::MAX);
                let repetition = build_repetition(&char_rule, min_len, max_len, "");
                let rule = format!("\"\\\"\\\" {} \\\"\\\"\" space", repetition);
                return self.add_rule(&rule_name, &rule);
            }
        } else if schema_type_str == Some("integer")
            && (schema.get("minimum").is_some()
                || schema.get("exclusiveMinimum").is_some()
                || schema.get("maximum").is_some()
                || schema.get("exclusiveMaximum").is_some())
        {
            let mut min_value = i32::MIN;
            let mut max_value = i32::MAX;
            if let Some(v) = schema.get("minimum").and_then(|v| v.as_i64()) {
                min_value = v as i32;
            } else if let Some(v) = schema.get("exclusiveMinimum").and_then(|v| v.as_i64()) {
                min_value = v as i32 + 1;
            }
            if let Some(v) = schema.get("maximum").and_then(|v| v.as_i64()) {
                max_value = v as i32;
            } else if let Some(v) = schema.get("exclusiveMaximum").and_then(|v| v.as_i64()) {
                max_value = v as i32 - 1;
            }
            let mut out = String::from("(");
            build_min_max_int(min_value, max_value, &mut out, 16, true);
            out.push_str(") space");
            return self.add_rule(&rule_name, &out);
        } else if schema.as_object().map(|m| m.is_empty()).unwrap_or(false)
            || schema_type_str == Some("object")
        {
            let rule = self.add_primitive("object", PRIMITIVE_RULES.get("object").unwrap());
            return self.add_rule(&rule_name, &rule);
        }

        if let Some(schema_type_str) = schema_type_str {
            if let Some(prim) = PRIMITIVE_RULES.get(schema_type_str) {
                return self.add_primitive(
                    if rule_name == "root" {
                        "root"
                    } else {
                        schema_type_str
                    },
                    prim,
                );
            }
        }

        self.errors
            .push(format!("Unrecognized schema: {}", schema.to_string()));
        String::new()
    }

    fn check_errors(&self) -> Result<(), SchemaError> {
        if !self.errors.is_empty() {
            return Err(SchemaError::Conversion(self.errors.join("\n")));
        }
        if !self.warnings.is_empty() {
            eprintln!(
                "WARNING: JSON schema conversion was incomplete: {}",
                self.warnings.join("; ")
            );
        }
        Ok(())
    }

    fn format_grammar(&self) -> String {
        let mut ss = String::new();
        for (name, rule) in &self.rules {
            ss.push_str(name);
            ss.push_str(" ::= ");
            ss.push_str(rule);
            ss.push('\n');
        }
        ss
    }
}

pub fn build_grammar<F>(cb: F, options: CommonGrammarOptions) -> Result<String, SchemaError>
where
    F: FnOnce(&CommonGrammarBuilder),
{
    let converter = Rc::new(RefCell::new(SchemaConverter::new(
        |_| Value::Null,
        options.dotall,
    )));
    let builder = CommonGrammarBuilder {
        add_rule: {
            let converter = Rc::clone(&converter);
            Box::new(move |name, rule| converter.borrow_mut().add_rule(name, rule))
        },
        add_schema: {
            let converter = Rc::clone(&converter);
            Box::new(move |name, schema| {
                let mut borrowed = converter.borrow_mut();
                borrowed.visit(schema, if name == "root" { "" } else { name })
            })
        },
        resolve_refs: {
            let converter = Rc::clone(&converter);
            Box::new(move |schema| converter.borrow_mut().resolve_refs(schema, ""))
        },
    };
    cb(&builder);
    {
        let borrowed = converter.borrow();
        borrowed.check_errors()?;
    }
    let grammar = converter.borrow().format_grammar();
    Ok(grammar)
}

pub fn json_schema_to_grammar(schema: &Value, force_gbnf: bool) -> Result<String, SchemaError> {
    #[cfg(feature = "llguidance")]
    {
        if !force_gbnf {
            return Ok(format!(
                "%llguidance {{}}\nstart: %json {}",
                schema.to_string()
            ));
        }
    }
    #[cfg(not(feature = "llguidance"))]
    {
        let _ = force_gbnf;
    }

    build_grammar(
        |builder| {
            let mut copy = schema.clone();
            builder.resolve_refs(&mut copy);
            builder.add_schema("", &copy);
        },
        CommonGrammarOptions::default(),
    )
}

pub fn json_schema_str_to_grammar(schema: &str, force_gbnf: bool) -> Result<String, SchemaError> {
    let value: Value = serde_json::from_str(schema)
        .map_err(|err| SchemaError::Conversion(format!("Failed to parse schema: {err}")))?;
    json_schema_to_grammar(&value, force_gbnf)
}
