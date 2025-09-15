use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    LookingForOpening,
    ThinkingStartedEatingWhitespace,
    Thinking,
    ThinkingDoneEatingWhitespace,
    ThinkingDone,
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct Parser {
    state: State,
    pub opening_tag: String,
    pub closing_tag: String,
    acc: String,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            state: State::LookingForOpening,
            opening_tag: String::new(),
            closing_tag: String::new(),
            acc: String::new(),
        }
    }
}

impl Parser {
    pub fn add_content(&mut self, content: &str) -> (String, String) {
        self.acc.push_str(content);
        let mut thinking = String::new();
        let mut remaining = String::new();
        let mut keep = true;
        while keep {
            let (t, r, k) = eat(self);
            thinking.push_str(&t);
            remaining.push_str(&r);
            keep = k;
        }
        (thinking, remaining)
    }

    pub fn state(&self) -> State {
        self.state
    }
}

fn eat(p: &mut Parser) -> (String, String, bool) {
    match p.state {
        State::LookingForOpening => {
            let trimmed = p.acc.trim_start().to_string();
            if trimmed.starts_with(&p.opening_tag) {
                let after = trimmed[p.opening_tag.len()..].trim_start().to_string();
                p.acc.clear();
                if after.is_empty() {
                    p.state = State::ThinkingStartedEatingWhitespace;
                } else {
                    p.state = State::Thinking;
                    p.acc.push_str(&after);
                }
                (String::new(), String::new(), true)
            } else if !p.opening_tag.is_empty() && p.opening_tag.starts_with(trimmed.as_str()) {
                (String::new(), String::new(), false)
            } else if trimmed.is_empty() {
                (String::new(), String::new(), false)
            } else {
                p.state = State::ThinkingDone;
                let untrimmed = std::mem::take(&mut p.acc);
                (String::new(), untrimmed, false)
            }
        }
        State::ThinkingStartedEatingWhitespace => {
            let trimmed = p.acc.trim_start().to_string();
            p.acc.clear();
            if trimmed.is_empty() {
                (String::new(), String::new(), false)
            } else {
                p.state = State::Thinking;
                p.acc.push_str(&trimmed);
                (String::new(), String::new(), true)
            }
        }
        State::Thinking => {
            if let Some(idx) = p.acc.find(&p.closing_tag) {
                let thinking = p.acc[..idx].to_string();
                let mut rem = p.acc[idx + p.closing_tag.len()..].to_string();
                rem = rem.trim_start().to_string();
                p.acc.clear();
                if rem.is_empty() {
                    p.state = State::ThinkingDoneEatingWhitespace;
                } else {
                    p.state = State::ThinkingDone;
                }
                (thinking, rem, false)
            } else if let Some(overlap) = overlap(&p.acc, &p.closing_tag) {
                let thinking = p.acc[..p.acc.len() - overlap].to_string();
                let rem = p.acc[p.acc.len() - overlap..].to_string();
                p.acc.clear();
                p.acc.push_str(&rem);
                (thinking, String::new(), false)
            } else {
                let acc = std::mem::take(&mut p.acc);
                (acc, String::new(), false)
            }
        }
        State::ThinkingDoneEatingWhitespace => {
            let trimmed = p.acc.trim_start().to_string();
            p.acc.clear();
            if trimmed.is_empty() {
                (String::new(), String::new(), false)
            } else {
                p.state = State::ThinkingDone;
                (String::new(), trimmed, false)
            }
        }
        State::ThinkingDone => {
            let acc = std::mem::take(&mut p.acc);
            (String::new(), acc, false)
        }
    }
}

fn overlap(s: &str, delim: &str) -> Option<usize> {
    if delim.is_empty() {
        return None;
    }
    let max = delim.len().min(s.len());
    for i in (1..=max).rev() {
        if s.ends_with(&delim[..i]) {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_extract() {
        let mut p = Parser {
            opening_tag: "<think>".into(),
            closing_tag: "</think>".into(),
            ..Default::default()
        };
        let (t, c) = p.add_content("<think>abc</think>def");
        assert_eq!(t, "abc");
        assert_eq!(c, "def");
    }
}
