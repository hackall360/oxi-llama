use std::cmp::min;
use std::collections::HashMap;

use api::Message;
use logutil::{error, trace, warn};
use nom::bytes::complete::{tag, take_until, take_while, take_while1};
use nom::character::complete::multispace0;
use nom::sequence::preceded;
use nom::IResult;
use unicode_ident::is_xid_continue;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarmonyParserState {
    LookingForMessageStart,
    ParsingHeader,
    ParsingContent,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HarmonyHeader {
    pub role: String,
    pub channel: String,
    pub recipient: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmonyEvent {
    MessageStart,
    HeaderComplete(HarmonyHeader),
    ContentEmitted(String),
    MessageEnd,
}

#[derive(Debug, Clone)]
pub struct HarmonyParser {
    state: HarmonyParserState,
    pub message_start_tag: String,
    pub message_end_tag: String,
    pub header_end_tag: String,
    acc: String,
    lifetime_acc: String,
}

impl Default for HarmonyParser {
    fn default() -> Self {
        Self {
            state: HarmonyParserState::LookingForMessageStart,
            message_start_tag: "<|start|>".to_string(),
            message_end_tag: "<|end|>".to_string(),
            header_end_tag: "<|message|>".to_string(),
            acc: String::new(),
            lifetime_acc: String::new(),
        }
    }
}

impl HarmonyParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_implicit_start(&mut self) {
        self.acc.push_str("<|start|>assistant");
    }

    pub fn add_implicit_start_or_prefill(&mut self, last_message: Option<&Message>) {
        if let Some(message) = last_message {
            if message.role == "assistant" {
                if !message.content.is_empty() {
                    self.acc
                        .push_str("<|start|>assistant<|channel|>final<|message|>");
                    return;
                } else if !message.thinking.is_empty() {
                    self.acc
                        .push_str("<|start|>assistant<|channel|>analysis<|message|>");
                    return;
                }
            }
        }
        self.add_implicit_start();
    }

    pub fn add_content(&mut self, content: &str) -> Vec<HarmonyEvent> {
        self.lifetime_acc.push_str(content);
        self.acc.push_str(content);

        let mut events = Vec::new();
        let mut keep_looping = true;
        while keep_looping {
            let (mut new_events, keep_eating) = self.eat();
            events.append(&mut new_events);
            keep_looping = keep_eating;
        }

        events
    }

    fn eat(&mut self) -> (Vec<HarmonyEvent>, bool) {
        match self.state {
            HarmonyParserState::LookingForMessageStart => {
                if let Some(idx) = self.acc.find(&self.message_start_tag) {
                    let before = &self.acc[..idx];
                    if !before.is_empty() {
                        warn!(content = %self.acc, "harmony parser: found message start tag in the middle of the content");
                    }
                    let after_index = idx + self.message_start_tag.len();
                    let after = self.acc[after_index..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    self.state = HarmonyParserState::ParsingHeader;
                    (vec![HarmonyEvent::MessageStart], true)
                } else {
                    (Vec::new(), false)
                }
            }
            HarmonyParserState::ParsingHeader => {
                if let Some(idx) = self.acc.find(&self.header_end_tag) {
                    let header = self.acc[..idx].to_string();
                    let after_index = idx + self.header_end_tag.len();
                    let after = self.acc[after_index..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    self.state = HarmonyParserState::ParsingContent;
                    let parsed_header = self.parse_header(&header);
                    (vec![HarmonyEvent::HeaderComplete(parsed_header)], true)
                } else {
                    (Vec::new(), false)
                }
            }
            HarmonyParserState::ParsingContent => {
                if let Some(idx) = self.acc.find(&self.message_end_tag) {
                    let content = self.acc[..idx].to_string();
                    let after_index = idx + self.message_end_tag.len();
                    let after = self.acc[after_index..].to_string();
                    self.acc.clear();
                    self.acc.push_str(&after);
                    self.state = HarmonyParserState::LookingForMessageStart;
                    let mut events = Vec::new();
                    if !content.is_empty() {
                        events.push(HarmonyEvent::ContentEmitted(content));
                    }
                    events.push(HarmonyEvent::MessageEnd);
                    (events, true)
                } else {
                    let overlap_len = overlap(&self.acc, &self.message_end_tag);
                    if overlap_len > 0 {
                        let split_point = self.acc.len() - overlap_len;
                        let content = self.acc[..split_point].to_string();
                        let remaining = self.acc[split_point..].to_string();
                        self.acc.clear();
                        self.acc.push_str(&remaining);
                        if content.is_empty() {
                            (Vec::new(), false)
                        } else {
                            (vec![HarmonyEvent::ContentEmitted(content)], false)
                        }
                    } else if self.acc.is_empty() {
                        (Vec::new(), false)
                    } else {
                        let content = std::mem::take(&mut self.acc);
                        (vec![HarmonyEvent::ContentEmitted(content)], false)
                    }
                }
            }
        }
    }

    fn parse_header(&self, raw: &str) -> HarmonyHeader {
        let mut harmony_header = HarmonyHeader::default();
        let mut working = raw.to_string();

        if working.contains("<|constrain|>") {
            working = working.replacen("<|constrain|>", " <|constrain|>", 1);
            working = working.trim().to_string();
        }

        let (channel, without_channel) = extract_channel(&working);
        if let Some(channel) = channel {
            harmony_header.channel = channel;
        }

        let tokens = tokenize(&without_channel);
        if tokens.is_empty() {
            error!(header = %without_channel, "harmony parser: missing role in header");
            return harmony_header;
        }

        let mut tokens = tokens;
        let first = tokens.remove(0);
        if first.starts_with("to=") {
            harmony_header.recipient = first[3..].to_string();
            harmony_header.role = "tool".to_string();
        } else {
            harmony_header.role = first;
        }

        if harmony_header.recipient.is_empty() {
            if let Some(recipient_token) = tokens.iter().find(|token| token.starts_with("to=")) {
                harmony_header.recipient = recipient_token[3..].to_string();
            }
        }

        harmony_header
    }
}

fn parse_channel_segment(input: &str) -> IResult<&str, (&str, &str)> {
    let (after_before, before) = take_until("<|channel|>")(input)?;
    let (after_tag, _) = tag("<|channel|>")(after_before)?;
    let (remaining, channel) = take_while(|c: char| !c.is_whitespace())(after_tag)?;
    Ok((remaining, (before, channel)))
}

fn extract_channel(raw: &str) -> (Option<String>, String) {
    match parse_channel_segment(raw) {
        Ok((remaining, (before, channel))) => {
            let mut rebuilt = String::with_capacity(before.len() + remaining.len());
            rebuilt.push_str(before);
            rebuilt.push_str(remaining);
            (Some(channel.to_string()), rebuilt.trim().to_string())
        }
        Err(_) => (None, raw.trim().to_string()),
    }
}

fn parse_token(input: &str) -> IResult<&str, &str> {
    preceded(multispace0, take_while1(|c: char| !c.is_whitespace()))(input)
}
fn tokenize(mut raw: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    loop {
        match parse_token(raw) {
            Ok((rest, token)) => {
                tokens.push(token.to_string());
                raw = rest;
            }
            Err(_) => break,
        }
    }
    tokens
}

fn overlap(s: &str, delim: &str) -> usize {
    let max = min(delim.len(), s.len());
    for i in (1..=max).rev() {
        if s.ends_with(&delim[..i]) {
            return i;
        }
    }
    0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarmonyMessageState {
    Normal,
    Thinking,
    ToolCalling,
}

pub struct HarmonyMessageHandler {
    state: HarmonyMessageState,
    pub harmony_parser: HarmonyParser,
    pub function_name_map: FunctionNameMap,
}

impl HarmonyMessageHandler {
    pub fn new() -> Self {
        Self {
            state: HarmonyMessageState::Normal,
            harmony_parser: HarmonyParser::new(),
            function_name_map: FunctionNameMap::new(),
        }
    }

    pub fn add_content(
        &mut self,
        content: &str,
        tool_parser: &mut HarmonyToolCallAccumulator,
    ) -> (String, String, String) {
        let mut content_sb = String::new();
        let mut thinking_sb = String::new();
        let mut tool_content_sb = String::new();

        let events = self.harmony_parser.add_content(content);
        for event in events {
            match event {
                HarmonyEvent::HeaderComplete(header) => {
                    trace!(role = %header.role, channel = %header.channel, recipient = %header.recipient, "harmony event header complete");
                    match header.channel.as_str() {
                        "analysis" => {
                            if !header.recipient.is_empty() {
                                self.state = HarmonyMessageState::ToolCalling;
                                tool_parser.set_tool_name(&header.recipient);
                            } else {
                                self.state = HarmonyMessageState::Thinking;
                            }
                        }
                        "commentary" => {
                            if !header.recipient.is_empty() {
                                self.state = HarmonyMessageState::ToolCalling;
                                tool_parser.set_tool_name(&header.recipient);
                            } else {
                                self.state = HarmonyMessageState::Normal;
                            }
                        }
                        "final" => {
                            self.state = HarmonyMessageState::Normal;
                        }
                        _ => {}
                    }
                }
                HarmonyEvent::ContentEmitted(emitted) => {
                    trace!(content = %emitted, state = ?self.state, "harmony event content");
                    match self.state {
                        HarmonyMessageState::Normal => content_sb.push_str(&emitted),
                        HarmonyMessageState::Thinking => thinking_sb.push_str(&emitted),
                        HarmonyMessageState::ToolCalling => tool_content_sb.push_str(&emitted),
                    }
                }
                HarmonyEvent::MessageEnd => {
                    self.state = HarmonyMessageState::Normal;
                }
                HarmonyEvent::MessageStart => {}
            }
        }

        (content_sb, thinking_sb, tool_content_sb)
    }

    pub fn create_tool_parser(&self) -> HarmonyToolCallAccumulator {
        HarmonyToolCallAccumulator {
            state: HarmonyToolCallState::Normal,
            acc: String::new(),
            current_tool_name: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarmonyToolCallState {
    Normal,
    ToolCalling,
}

pub struct HarmonyToolCallAccumulator {
    state: HarmonyToolCallState,
    acc: String,
    current_tool_name: Option<String>,
}

impl HarmonyToolCallAccumulator {
    pub fn set_tool_name(&mut self, tool_name: &str) {
        self.current_tool_name = Some(tool_name.to_string());
        self.state = HarmonyToolCallState::ToolCalling;
    }

    pub fn add(&mut self, content: &str) {
        self.acc.push_str(content);
    }

    pub fn drain(&mut self) -> (Option<String>, String) {
        let content = std::mem::take(&mut self.acc);
        self.state = HarmonyToolCallState::Normal;
        (self.current_tool_name.clone(), content)
    }

    pub fn content(&self) -> &str {
        &self.acc
    }
}

#[derive(Debug, Default, Clone)]
pub struct FunctionNameMap {
    user_to_harmony: HashMap<String, String>,
    harmony_to_user: HashMap<String, String>,
}

impl FunctionNameMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn convert_and_add(&mut self, user_function_name: &str) -> String {
        let harmony_function_name = self.derive_name(user_function_name);
        self.user_to_harmony.insert(
            user_function_name.to_string(),
            harmony_function_name.clone(),
        );
        self.harmony_to_user.insert(
            harmony_function_name.clone(),
            user_function_name.to_string(),
        );
        harmony_function_name
    }

    pub fn original_from_converted(&self, harmony_function_name: &str) -> String {
        self.harmony_to_user
            .get(harmony_function_name)
            .cloned()
            .unwrap_or_else(|| {
                warn!(
                    harmony_function_name = harmony_function_name,
                    "harmony parser: no reverse mapping found for function name"
                );
                harmony_function_name.to_string()
            })
    }

    fn derive_name(&mut self, user_function_name: &str) -> String {
        let original_candidate = self.convert_to_valid_chars(user_function_name);
        let mut candidate = original_candidate.clone();
        let mut count = 2;
        while self.harmony_to_user.contains_key(&candidate) {
            candidate = format!("{original_candidate}_{count}");
            count += 1;
        }
        candidate
    }

    fn convert_to_valid_chars(&self, user_function_name: &str) -> String {
        let mut candidate: String = user_function_name
            .chars()
            .filter_map(|r| match r {
                ' ' | '-' | '.' => Some('_'),
                '_' | '$' => Some(r),
                _ if is_xid_continue(r) => Some(r),
                _ => None,
            })
            .collect();

        if candidate.is_empty() {
            return "unnamed".to_string();
        }

        if candidate
            .chars()
            .next()
            .map(|c| c.is_numeric())
            .unwrap_or(false)
        {
            candidate = format!("_{candidate}");
        }

        candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(role: &str, channel: &str, recipient: &str) -> HarmonyHeader {
        HarmonyHeader {
            role: role.to_string(),
            channel: channel.to_string(),
            recipient: recipient.to_string(),
        }
    }

    #[test]
    fn header_parsing() {
        struct Case {
            input: &'static str,
            want_role: &'static str,
            want_channel: &'static str,
            want_recipient: &'static str,
        }

        let cases = vec![
            Case {
                input: "assistant<|channel|>analysis",
                want_role: "assistant",
                want_channel: "analysis",
                want_recipient: "",
            },
            Case {
                input: "assistant<|channel|>analysis to=functions.get_weather",
                want_role: "assistant",
                want_channel: "analysis",
                want_recipient: "functions.get_weather",
            },
            Case {
                input: "assistant to=functions.get_weather<|channel|>analysis",
                want_role: "assistant",
                want_channel: "analysis",
                want_recipient: "functions.get_weather",
            },
            Case {
                input: "to=functions.get_weather<|channel|>analysis",
                want_role: "tool",
                want_channel: "analysis",
                want_recipient: "functions.get_weather",
            },
            Case {
                input: "assistant to=functions.get_weather abc<|channel|>analysis",
                want_role: "assistant",
                want_channel: "analysis",
                want_recipient: "functions.get_weather",
            },
            Case {
                input: "assistant<|channel|>commentary to=functions.get_weather <|constrain|>json",
                want_role: "assistant",
                want_channel: "commentary",
                want_recipient: "functions.get_weather",
            },
            Case {
                input: "assistant to=functions.get_weather<|channel|>commentary <|constrain|>json",
                want_role: "assistant",
                want_channel: "commentary",
                want_recipient: "functions.get_weather",
            },
            Case {
                input: "assistant<|channel|>commentary to=functions.get_weather<|constrain|>json",
                want_role: "assistant",
                want_channel: "commentary",
                want_recipient: "functions.get_weather",
            },
            Case {
                input: "assistant to=functions.get_weather<|channel|>commentary<|constrain|>json",
                want_role: "assistant",
                want_channel: "commentary",
                want_recipient: "functions.get_weather",
            },
        ];

        for (i, case) in cases.into_iter().enumerate() {
            let parser = HarmonyParser::new();
            let header = parser.parse_header(case.input);
            assert_eq!(header.role, case.want_role, "case {i} role");
            assert_eq!(header.channel, case.want_channel, "case {i} channel");
            assert_eq!(header.recipient, case.want_recipient, "case {i} recipient");
        }
    }

    #[test]
    fn harmony_parser_header_event() {
        struct Case {
            input: &'static str,
            want_role: &'static str,
            want_channel: &'static str,
            want_recipient: &'static str,
            implicit_start: bool,
        }

        let cases = vec![
            Case {
                input: "<|start|>user<|message|>What is 2 + 2?<|end|>",
                want_role: "user",
                want_channel: "",
                want_recipient: "",
                implicit_start: false,
            },
            Case {
                input: "<|start|>assistant<|channel|>analysis<|message|>What is 2 + 2?<|end|>",
                want_role: "assistant",
                want_channel: "analysis",
                want_recipient: "",
                implicit_start: false,
            },
            Case {
                input: "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"San Francisco\"}<|call|><|start|>functions.get_weather to=assistant<|message|>{\"sunny\": true, \"temperature\": 20}<|end|>",
                want_role: "assistant",
                want_channel: "commentary",
                want_recipient: "functions.get_weather",
                implicit_start: false,
            },
            Case {
                input: "<|channel|>analysis<|message|>User asks weather in SF. We need location. Use get_current_weather with location \"San Francisco, CA\".<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{\"location\":\"San Francisco, CA\"}<|call|>",
                want_role: "assistant",
                want_channel: "analysis",
                want_recipient: "",
                implicit_start: true,
            },
        ];

        for (i, case) in cases.into_iter().enumerate() {
            let mut parser = HarmonyParser::new();
            if case.implicit_start {
                parser.add_implicit_start();
            }
            let events = parser.add_content(case.input);
            assert!(!events.is_empty(), "case {i}: expected at least one event");

            let header_event = events.iter().find_map(|event| {
                if let HarmonyEvent::HeaderComplete(header) = event {
                    Some(header.clone())
                } else {
                    None
                }
            });

            let Some(header) = header_event else {
                panic!("case {i}: expected a header event");
            };

            assert_eq!(header.role, case.want_role, "case {i} role");
            assert_eq!(header.channel, case.want_channel, "case {i} channel");
            assert_eq!(header.recipient, case.want_recipient, "case {i} recipient");
        }
    }

    #[test]
    fn harmony_parser_non_streaming() {
        struct Case {
            input: &'static str,
            implicit_start: bool,
            want_events: Vec<HarmonyEvent>,
        }

        let cases = vec![
            Case {
                input: "<|start|>user<|message|>What is 2 + 2?<|end|>",
                implicit_start: false,
                want_events: vec![
                    HarmonyEvent::MessageStart,
                    HarmonyEvent::HeaderComplete(header("user", "", "")),
                    HarmonyEvent::ContentEmitted("What is 2 + 2?".to_string()),
                    HarmonyEvent::MessageEnd,
                ],
            },
            Case {
                input: "<|start|>assistant<|channel|>analysis<|message|>The answer is 4<|end|>",
                implicit_start: false,
                want_events: vec![
                    HarmonyEvent::MessageStart,
                    HarmonyEvent::HeaderComplete(header("assistant", "analysis", "")),
                    HarmonyEvent::ContentEmitted("The answer is 4".to_string()),
                    HarmonyEvent::MessageEnd,
                ],
            },
            Case {
                input: "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>Computing...<|end|>",
                implicit_start: false,
                want_events: vec![
                    HarmonyEvent::MessageStart,
                    HarmonyEvent::HeaderComplete(header("assistant", "commentary", "functions.calc")),
                    HarmonyEvent::ContentEmitted("Computing...".to_string()),
                    HarmonyEvent::MessageEnd,
                ],
            },
            Case {
                input: "<|start|>user<|message|><|end|>",
                implicit_start: false,
                want_events: vec![
                    HarmonyEvent::MessageStart,
                    HarmonyEvent::HeaderComplete(header("user", "", "")),
                    HarmonyEvent::MessageEnd,
                ],
            },
            Case {
                input: "<|start|>user<|message|>Hello<|end|><|start|>assistant<|message|>Hi!<|end|>",
                implicit_start: false,
                want_events: vec![
                    HarmonyEvent::MessageStart,
                    HarmonyEvent::HeaderComplete(header("user", "", "")),
                    HarmonyEvent::ContentEmitted("Hello".to_string()),
                    HarmonyEvent::MessageEnd,
                    HarmonyEvent::MessageStart,
                    HarmonyEvent::HeaderComplete(header("assistant", "", "")),
                    HarmonyEvent::ContentEmitted("Hi!".to_string()),
                    HarmonyEvent::MessageEnd,
                ],
            },
            Case {
                input: "<|channel|>analysis<|message|>Thinking about the request<|end|>",
                implicit_start: true,
                want_events: vec![
                    HarmonyEvent::MessageStart,
                    HarmonyEvent::HeaderComplete(header("assistant", "analysis", "")),
                    HarmonyEvent::ContentEmitted("Thinking about the request".to_string()),
                    HarmonyEvent::MessageEnd,
                ],
            },
        ];

        for (i, case) in cases.into_iter().enumerate() {
            let mut parser = HarmonyParser::new();
            if case.implicit_start {
                parser.add_implicit_start();
            }
            let events = parser.add_content(case.input);
            assert_eq!(events, case.want_events, "case {i}");
        }
    }

    #[test]
    fn harmony_parser_streaming() {
        struct Step {
            input: &'static str,
            want_events: Vec<HarmonyEvent>,
        }

        struct Case {
            desc: &'static str,
            implicit_start: bool,
            steps: Vec<Step>,
        }

        let cases = vec![
            Case {
                desc: "simple message streamed character by character",
                implicit_start: false,
                steps: vec![
                    Step {
                        input: "<",
                        want_events: vec![],
                    },
                    Step {
                        input: "|",
                        want_events: vec![],
                    },
                    Step {
                        input: "start|>u",
                        want_events: vec![HarmonyEvent::MessageStart],
                    },
                    Step {
                        input: "ser<|mess",
                        want_events: vec![],
                    },
                    Step {
                        input: "age|>Hi",
                        want_events: vec![
                            HarmonyEvent::HeaderComplete(header("user", "", "")),
                            HarmonyEvent::ContentEmitted("Hi".to_string()),
                        ],
                    },
                    Step {
                        input: " there",
                        want_events: vec![HarmonyEvent::ContentEmitted(" there".to_string())],
                    },
                    Step {
                        input: "<|e",
                        want_events: vec![],
                    },
                    Step {
                        input: "nd|>",
                        want_events: vec![HarmonyEvent::MessageEnd],
                    },
                ],
            },
            Case {
                desc: "message with channel streamed",
                implicit_start: false,
                steps: vec![
                    Step {
                        input: "<|start|>assistant",
                        want_events: vec![HarmonyEvent::MessageStart],
                    },
                    Step {
                        input: "<|chan",
                        want_events: vec![],
                    },
                    Step {
                        input: "nel|>analysis",
                        want_events: vec![],
                    },
                    Step {
                        input: "<|message|>",
                        want_events: vec![HarmonyEvent::HeaderComplete(header(
                            "assistant",
                            "analysis",
                            "",
                        ))],
                    },
                    Step {
                        input: "Thinking",
                        want_events: vec![HarmonyEvent::ContentEmitted("Thinking".to_string())],
                    },
                    Step {
                        input: "...",
                        want_events: vec![HarmonyEvent::ContentEmitted("...".to_string())],
                    },
                    Step {
                        input: "<|end|>",
                        want_events: vec![HarmonyEvent::MessageEnd],
                    },
                ],
            },
            Case {
                desc: "message with channel and recipient",
                implicit_start: false,
                steps: vec![
                    Step {
                        input:
                            "<|start|>assistant<|channel|>commentary to=functions.calc<|message|>",
                        want_events: vec![
                            HarmonyEvent::MessageStart,
                            HarmonyEvent::HeaderComplete(header(
                                "assistant",
                                "commentary",
                                "functions.calc",
                            )),
                        ],
                    },
                    Step {
                        input: "{\"x\": 5}",
                        want_events: vec![HarmonyEvent::ContentEmitted("{\"x\": 5}".to_string())],
                    },
                    Step {
                        input: "<|end|>",
                        want_events: vec![HarmonyEvent::MessageEnd],
                    },
                ],
            },
            Case {
                desc: "message with channel and recipient (receipient before channel)",
                implicit_start: false,
                steps: vec![
                    Step {
                        input:
                            "<|start|>assistant to=functions.calc<|channel|>commentary<|message|>",
                        want_events: vec![
                            HarmonyEvent::MessageStart,
                            HarmonyEvent::HeaderComplete(header(
                                "assistant",
                                "commentary",
                                "functions.calc",
                            )),
                        ],
                    },
                    Step {
                        input: "{\"x\": 5}",
                        want_events: vec![HarmonyEvent::ContentEmitted("{\"x\": 5}".to_string())],
                    },
                    Step {
                        input: "<|end|>",
                        want_events: vec![HarmonyEvent::MessageEnd],
                    },
                ],
            },
            Case {
                desc: "implicit start with channel",
                implicit_start: true,
                steps: vec![
                    Step {
                        input: "<|channel|>thinking",
                        want_events: vec![HarmonyEvent::MessageStart],
                    },
                    Step {
                        input: "<|message|>",
                        want_events: vec![HarmonyEvent::HeaderComplete(header(
                            "assistant",
                            "thinking",
                            "",
                        ))],
                    },
                    Step {
                        input: "Processing request",
                        want_events: vec![HarmonyEvent::ContentEmitted(
                            "Processing request".to_string(),
                        )],
                    },
                    Step {
                        input: "<|end|>",
                        want_events: vec![HarmonyEvent::MessageEnd],
                    },
                ],
            },
            Case {
                desc: "multiple messages streamed",
                implicit_start: false,
                steps: vec![
                    Step {
                        input: "<|start|>user<|message|>Hello<|end|>",
                        want_events: vec![
                            HarmonyEvent::MessageStart,
                            HarmonyEvent::HeaderComplete(header("user", "", "")),
                            HarmonyEvent::ContentEmitted("Hello".to_string()),
                            HarmonyEvent::MessageEnd,
                        ],
                    },
                    Step {
                        input: "<|start|>",
                        want_events: vec![HarmonyEvent::MessageStart],
                    },
                    Step {
                        input: "assistant<|message|>",
                        want_events: vec![HarmonyEvent::HeaderComplete(header(
                            "assistant",
                            "",
                            "",
                        ))],
                    },
                    Step {
                        input: "Hi!",
                        want_events: vec![HarmonyEvent::ContentEmitted("Hi!".to_string())],
                    },
                    Step {
                        input: "<|end|>",
                        want_events: vec![HarmonyEvent::MessageEnd],
                    },
                ],
            },
            Case {
                desc: "empty message",
                implicit_start: false,
                steps: vec![Step {
                    input: "<|start|>system<|message|><|end|>",
                    want_events: vec![
                        HarmonyEvent::MessageStart,
                        HarmonyEvent::HeaderComplete(header("system", "", "")),
                        HarmonyEvent::MessageEnd,
                    ],
                }],
            },
            Case {
                desc: "partial tag that looks like end but isn't",
                implicit_start: false,
                steps: vec![
                    Step {
                        input: "<|start|>user<|message|>test<|e",
                        want_events: vec![
                            HarmonyEvent::MessageStart,
                            HarmonyEvent::HeaderComplete(header("user", "", "")),
                            HarmonyEvent::ContentEmitted("test".to_string()),
                        ],
                    },
                    Step {
                        input: "xample|>more",
                        want_events: vec![HarmonyEvent::ContentEmitted(
                            "<|example|>more".to_string(),
                        )],
                    },
                    Step {
                        input: "<|end|>",
                        want_events: vec![HarmonyEvent::MessageEnd],
                    },
                ],
            },
        ];

        for case in cases {
            let mut parser = HarmonyParser::new();
            if case.implicit_start {
                parser.add_implicit_start();
            }
            for (i, step) in case.steps.iter().enumerate() {
                let events = parser.add_content(step.input);
                assert_eq!(events, step.want_events, "{} step {i}", case.desc);
            }
        }
    }

    #[test]
    fn function_convert_to_valid_chars() {
        struct Case {
            name: &'static str,
            input: &'static str,
            want: &'static str,
        }

        let cases = vec![
            Case {
                name: "replace spaces with underscores",
                input: "get weather",
                want: "get_weather",
            },
            Case {
                name: "replace hyphens with underscores",
                input: "get-weather",
                want: "get_weather",
            },
            Case {
                name: "replace periods with underscores",
                input: "get.weather",
                want: "get_weather",
            },
            Case {
                name: "disallow non-word characters",
                input: "get weather!",
                want: "get_weather",
            },
            Case {
                name: "strip out invalid non-alphanumeric unicode characters",
                input: "a🫠bc",
                want: "abc",
            },
            Case {
                name: "names that only contain invalid characters",
                input: "🫠",
                want: "unnamed",
            },
            Case {
                name: "leading number",
                input: "123",
                want: "_123",
            },
            Case {
                name: "$ allowed",
                input: "$",
                want: "$",
            },
            Case {
                name: "allow weird unicode letter characters",
                input: "𝓸𝓵𝓵𝓪𝓶𝓪",
                want: "𝓸𝓵𝓵𝓪𝓶𝓪",
            },
            Case {
                name: "disallow non-word characters that look like words",
                input: "ⓞⓛⓛⓐⓜⓐ123",
                want: "_123",
            },
        ];

        for (i, case) in cases.into_iter().enumerate() {
            let map = FunctionNameMap::new();
            let got = map.convert_to_valid_chars(case.input);
            assert_eq!(got, case.want, "case {i} {}", case.name);
        }
    }

    #[test]
    fn function_convert_and_add() {
        struct Case {
            name: &'static str,
            input: Vec<&'static str>,
            want: Vec<&'static str>,
        }

        let cases = vec![
            Case {
                name: "basic dupe handling",
                input: vec!["get weather", "get weather"],
                want: vec!["get_weather", "get_weather_2"],
            },
            Case {
                name: "dupes from different user-specified names",
                input: vec!["get weather", "get_weather", "get-weather"],
                want: vec!["get_weather", "get_weather_2", "get_weather_3"],
            },
            Case {
                name: "non dupes after dupes",
                input: vec![
                    "get weather",
                    "get_weather",
                    "get-weather",
                    "something-different",
                ],
                want: vec![
                    "get_weather",
                    "get_weather_2",
                    "get_weather_3",
                    "something_different",
                ],
            },
            Case {
                name: "multiple sets of dupes",
                input: vec!["a", "a", "b", "a", "a", "b", "a"],
                want: vec!["a", "a_2", "b", "a_3", "a_4", "b_2", "a_5"],
            },
        ];

        for case in cases {
            let mut map = FunctionNameMap::new();
            for (idx, input) in case.input.iter().enumerate() {
                let got = map.convert_and_add(input);
                let want = case.want[idx];
                assert_eq!(got, want, "{} index {idx}", case.name);
                assert_eq!(map.user_to_harmony.get(*input).unwrap(), want);
                assert_eq!(map.harmony_to_user.get(want).unwrap(), input);
            }
        }
    }
}
