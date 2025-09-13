use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
#[cfg(test)]
use std::collections::VecDeque;
use std::io;

#[derive(Clone, Debug)]
pub struct Prompt {
    pub prompt: String,
    pub alt_prompt: String,
    pub placeholder: String,
    pub alt_placeholder: String,
    pub use_alt: bool,
}

impl Default for Prompt {
    fn default() -> Self {
        Self {
            prompt: ">".into(),
            alt_prompt: ">".into(),
            placeholder: String::new(),
            alt_placeholder: String::new(),
            use_alt: false,
        }
    }
}

impl Prompt {
    pub fn prompt(&self) -> &str {
        if self.use_alt {
            &self.alt_prompt
        } else {
            &self.prompt
        }
    }

    pub fn placeholder(&self) -> &str {
        if self.use_alt {
            &self.alt_placeholder
        } else {
            &self.placeholder
        }
    }
}

#[derive(Debug)]
pub enum ReadlineError {
    Interrupted,
    Eof,
    Io(io::Error),
}

impl PartialEq for ReadlineError {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (ReadlineError::Interrupted, ReadlineError::Interrupted)
                | (ReadlineError::Eof, ReadlineError::Eof)
        )
    }
}

impl std::fmt::Display for ReadlineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadlineError::Interrupted => write!(f, "interrupted"),
            ReadlineError::Eof => write!(f, "eof"),
            ReadlineError::Io(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for ReadlineError {}

pub trait Terminal {
    fn read(&mut self) -> io::Result<Event>;
}

pub struct CrosstermTerminal;

impl Terminal for CrosstermTerminal {
    fn read(&mut self) -> io::Result<Event> {
        event::read()
    }
}

#[cfg(test)]
pub struct MockTerminal {
    events: VecDeque<Event>,
}

#[cfg(test)]
impl MockTerminal {
    pub fn new(events: Vec<Event>) -> Self {
        Self {
            events: events.into(),
        }
    }

    pub fn push(&mut self, events: Vec<Event>) {
        self.events.extend(events);
    }
}

#[cfg(test)]
impl Terminal for MockTerminal {
    fn read(&mut self) -> io::Result<Event> {
        if let Some(ev) = self.events.pop_front() {
            Ok(ev)
        } else {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "no events"))
        }
    }
}

#[derive(Default, Debug)]
pub struct Buffer {
    buf: Vec<char>,
    pos: usize,
}

impl Buffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn add(&mut self, c: char) {
        self.buf.insert(self.pos, c);
        self.pos += 1;
    }

    pub fn remove(&mut self) {
        if self.pos > 0 {
            self.pos -= 1;
            self.buf.remove(self.pos);
        }
    }

    pub fn delete(&mut self) {
        if self.pos < self.buf.len() {
            self.buf.remove(self.pos);
        }
    }

    pub fn move_left(&mut self) {
        if self.pos > 0 {
            self.pos -= 1;
        }
    }

    pub fn move_right(&mut self) {
        if self.pos < self.buf.len() {
            self.pos += 1;
        }
    }

    pub fn move_to_start(&mut self) {
        self.pos = 0;
    }

    pub fn move_to_end(&mut self) {
        self.pos = self.buf.len();
    }

    pub fn replace(&mut self, chars: &[char]) {
        self.buf = chars.to_vec();
        self.pos = self.buf.len();
    }

    pub fn string(&self) -> String {
        self.buf.iter().collect()
    }
}

#[derive(Default, Debug)]
pub struct History {
    entries: Vec<String>,
    pos: usize,
    pub enabled: bool,
}

impl History {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            pos: 0,
            enabled: true,
        }
    }

    pub fn add(&mut self, s: String) {
        if self.enabled {
            self.entries.push(s);
            self.pos = self.entries.len();
        }
    }

    pub fn prev(&mut self) -> Option<String> {
        if self.pos > 0 {
            self.pos -= 1;
            Some(self.entries[self.pos].clone())
        } else {
            None
        }
    }

    pub fn next(&mut self) -> Option<String> {
        if self.pos < self.entries.len() {
            self.pos += 1;
            if self.pos == self.entries.len() {
                None
            } else {
                Some(self.entries[self.pos].clone())
            }
        } else {
            None
        }
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

pub struct Instance<T: Terminal> {
    pub prompt: Prompt,
    pub terminal: T,
    pub history: History,
    pub pasting: bool,
}

impl<T: Terminal> Instance<T> {
    pub fn new(prompt: Prompt, terminal: T) -> Self {
        Self {
            prompt,
            terminal,
            history: History::new(),
            pasting: false,
        }
    }

    pub fn readline(&mut self) -> Result<String, ReadlineError> {
        let mut buf = Buffer::new();
        loop {
            let ev = self.terminal.read().map_err(ReadlineError::Io)?;
            if let Event::Key(KeyEvent {
                code, modifiers, ..
            }) = ev
            {
                match (code, modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                        return Err(ReadlineError::Interrupted)
                    }
                    (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                        if buf.is_empty() {
                            return Err(ReadlineError::Eof);
                        }
                    }
                    (KeyCode::Char(ch), mods) if mods.is_empty() => {
                        buf.add(ch);
                    }
                    (KeyCode::Backspace, _) => {
                        buf.remove();
                    }
                    (KeyCode::Delete, _) => {
                        if buf.is_empty() {
                            return Err(ReadlineError::Eof);
                        }
                        buf.delete();
                    }
                    (KeyCode::Left, _) => buf.move_left(),
                    (KeyCode::Right, _) => buf.move_right(),
                    (KeyCode::Up, _) => {
                        if let Some(line) = self.history.prev() {
                            let chars: Vec<char> = line.chars().collect();
                            buf.replace(&chars);
                        }
                    }
                    (KeyCode::Down, _) => {
                        if let Some(line) = self.history.next() {
                            let chars: Vec<char> = line.chars().collect();
                            buf.replace(&chars);
                        } else {
                            buf.replace(&[]);
                        }
                    }
                    (KeyCode::Enter, _) => {
                        let line = buf.string();
                        if !line.is_empty() {
                            self.history.add(line.clone());
                        }
                        return Ok(line);
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn history_enable(&mut self) {
        self.history.enable();
    }

    pub fn history_disable(&mut self) {
        self.history.disable();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev_char(c: char) -> Event {
        Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
    }
    fn ev_ctrl(c: char) -> Event {
        Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL))
    }
    fn ev(code: KeyCode) -> Event {
        Event::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    #[test]
    fn editing_basic() {
        let events = vec![
            ev_char('a'),
            ev(KeyCode::Left),
            ev_char('b'),
            ev(KeyCode::Enter),
        ];
        let term = MockTerminal::new(events);
        let prompt = Prompt::default();
        let mut inst = Instance::new(prompt, term);
        let line = inst.readline().unwrap();
        assert_eq!(line, "ba");
    }

    #[test]
    fn history_navigation() {
        let mut term = MockTerminal::new(vec![]);
        term.push(vec![
            ev_char('f'),
            ev_char('o'),
            ev_char('o'),
            ev(KeyCode::Enter),
        ]);
        let prompt = Prompt::default();
        let mut inst = Instance::new(prompt, term);
        let line1 = inst.readline().unwrap();
        assert_eq!(line1, "foo");
        inst.terminal
            .push(vec![ev(KeyCode::Up), ev(KeyCode::Enter)]);
        let line2 = inst.readline().unwrap();
        assert_eq!(line2, "foo");
    }

    #[test]
    fn ctrl_c_interrupt() {
        let events = vec![ev_ctrl('c')];
        let term = MockTerminal::new(events);
        let prompt = Prompt::default();
        let mut inst = Instance::new(prompt, term);
        let err = inst.readline().unwrap_err();
        assert_eq!(err, ReadlineError::Interrupted);
    }

    #[test]
    fn ctrl_d_eof() {
        let events = vec![ev_ctrl('d')];
        let term = MockTerminal::new(events);
        let prompt = Prompt::default();
        let mut inst = Instance::new(prompt, term);
        let err = inst.readline().unwrap_err();
        assert_eq!(err, ReadlineError::Eof);
    }
}
