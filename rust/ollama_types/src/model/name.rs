use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{PathBuf, MAIN_SEPARATOR};

pub const MISSING_PART: &str = "!MISSING!";
const DEFAULT_HOST: &str = "registry.ollama.ai";
const DEFAULT_NAMESPACE: &str = "library";
const DEFAULT_TAG: &str = "latest";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Name {
    #[serde(default)]
    pub host: String,
    #[serde(default)]
    pub namespace: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub tag: String,
}

pub fn default_name() -> Name {
    Name {
        host: DEFAULT_HOST.to_string(),
        namespace: DEFAULT_NAMESPACE.to_string(),
        model: String::new(),
        tag: DEFAULT_TAG.to_string(),
    }
}

pub fn merge(mut a: Name, b: Name) -> Name {
    if a.host.is_empty() {
        a.host = b.host;
    }
    if a.namespace.is_empty() {
        a.namespace = b.namespace;
    }
    if a.tag.is_empty() {
        a.tag = b.tag;
    }
    a
}

pub fn parse_name(s: &str) -> Name {
    merge(parse_name_bare(s), default_name())
}

pub fn parse_name_bare(s: &str) -> Name {
    let mut n = Name::default();
    let mut s = s.to_string();

    if let (Some(colon), Some(slash)) = (s.rfind(':'), s.rfind('/')) {
        if colon > slash {
            let (before, after, _) = cut_promised(&s, ':');
            s = before.to_string();
            n.tag = after.to_string();
        }
    } else if s.contains(':') {
        let (before, after, _) = cut_promised(&s, ':');
        s = before.to_string();
        n.tag = after.to_string();
    }

    let (before, model, promised) = cut_promised(&s, '/');
    if !promised {
        n.model = s;
        return n;
    }
    n.model = model.to_string();
    s = before.to_string();

    let (before, namespace, promised) = cut_promised(&s, '/');
    if !promised {
        n.namespace = s;
        return n;
    }
    n.namespace = namespace.to_string();
    s = before.to_string();

    if let Some(idx) = s.find("//") {
        let after = &s[idx + 2..];
        n.host = after.to_string();
    } else {
        n.host = s;
    }
    n
}

pub fn parse_name_from_filepath(s: &str) -> Name {
    let parts: Vec<&str> = s.split(MAIN_SEPARATOR).collect();
    if parts.len() != 4 {
        return Name::default();
    }
    let n = Name {
        host: parts[0].to_string(),
        namespace: parts[1].to_string(),
        model: parts[2].to_string(),
        tag: parts[3].to_string(),
    };
    if !n.is_fully_qualified() {
        Name::default()
    } else {
        n
    }
}

impl Name {
    pub fn is_valid(&self) -> bool {
        self.is_fully_qualified()
    }

    pub fn is_fully_qualified(&self) -> bool {
        let parts = [
            (&self.host, PartKind::Host),
            (&self.namespace, PartKind::Namespace),
            (&self.model, PartKind::Model),
            (&self.tag, PartKind::Tag),
        ];
        for (part, kind) in parts {
            if !is_valid_part(kind, part) {
                return false;
            }
        }
        true
    }

    pub fn filepath(&self) -> PathBuf {
        if !self.is_fully_qualified() {
            panic!("illegal attempt to get filepath of invalid name");
        }
        PathBuf::from(&self.host)
            .join(&self.namespace)
            .join(&self.model)
            .join(&self.tag)
    }

    pub fn display_shortest(&self) -> String {
        let mut out = String::new();
        if !self.host.eq_ignore_ascii_case(DEFAULT_HOST) {
            out.push_str(&self.host);
            out.push('/');
            out.push_str(&self.namespace);
            out.push('/');
        } else if !self.namespace.eq_ignore_ascii_case(DEFAULT_NAMESPACE) {
            out.push_str(&self.namespace);
            out.push('/');
        }
        out.push_str(&self.model);
        out.push(':');
        out.push_str(&self.tag);
        out
    }

    pub fn equal_fold(&self, other: &Name) -> bool {
        self.host.eq_ignore_ascii_case(&other.host)
            && self.namespace.eq_ignore_ascii_case(&other.namespace)
            && self.model.eq_ignore_ascii_case(&other.model)
            && self.tag.eq_ignore_ascii_case(&other.tag)
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b = String::new();
        if !self.host.is_empty() {
            b.push_str(&self.host);
            b.push('/');
        }
        if !self.namespace.is_empty() {
            b.push_str(&self.namespace);
            b.push('/');
        }
        b.push_str(&self.model);
        if !self.tag.is_empty() {
            b.push(':');
            b.push_str(&self.tag);
        }
        write!(f, "{}", b)
    }
}

#[derive(Copy, Clone, Debug)]
enum PartKind {
    Host,
    Namespace,
    Model,
    Tag,
    Digest,
}

pub fn is_valid_namespace(s: &str) -> bool {
    is_valid_part(PartKind::Namespace, s)
}

fn is_valid_len(kind: PartKind, s: &str) -> bool {
    match kind {
        PartKind::Host => (1..=350).contains(&s.len()),
        PartKind::Tag => (1..=80).contains(&s.len()),
        _ => (1..=80).contains(&s.len()),
    }
}

fn is_valid_part(kind: PartKind, s: &str) -> bool {
    if !is_valid_len(kind, s) {
        return false;
    }
    for (i, c) in s.bytes().enumerate() {
        if i == 0 {
            if !is_alphanumeric_or_underscore(c) {
                return false;
            }
            continue;
        }
        match c {
            b'_' | b'-' => {}
            b'.' => {
                if matches!(kind, PartKind::Namespace) {
                    return false;
                }
            }
            b':' => {
                if !matches!(kind, PartKind::Host | PartKind::Digest) {
                    return false;
                }
            }
            _ => {
                if !is_alphanumeric_or_underscore(c) {
                    return false;
                }
            }
        }
    }
    true
}

fn is_alphanumeric_or_underscore(c: u8) -> bool {
    (b'A'..=b'Z').contains(&c)
        || (b'a'..=b'z').contains(&c)
        || (b'0'..=b'9').contains(&c)
        || c == b'_'
}

fn cut_last(s: &str, sep: char) -> (String, String, bool) {
    if let Some(i) = s.rfind(sep) {
        let before = &s[..i];
        let after = &s[i + sep.len_utf8()..];
        (before.to_string(), after.to_string(), true)
    } else {
        (s.to_string(), String::new(), false)
    }
}

fn cut_promised(s: &str, sep: char) -> (String, String, bool) {
    let (before, after, ok) = cut_last(s, sep);
    if !ok {
        return (before, after, false);
    }
    (
        if before.is_empty() {
            MISSING_PART.to_string()
        } else {
            before
        },
        if after.is_empty() {
            MISSING_PART.to_string()
        } else {
            after
        },
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_valid_part_cases() {
        let cases = [
            (PartKind::Host, "", false),
            (PartKind::Host, "a", true),
            (PartKind::Host, "a.", true),
            (PartKind::Host, "a.b", true),
            (PartKind::Host, "a:123", true),
            (PartKind::Host, "a:123/aa/bb", false),
            (PartKind::Namespace, "bb", true),
            (PartKind::Namespace, "a.", false),
            (PartKind::Model, "-h", false),
            (
                PartKind::Digest,
                "sha256-1000000000000000000000000000000000000000000000000000000000000000",
                true,
            ),
        ];

        for (kind, s, want) in cases {
            assert_eq!(is_valid_part(kind, s), want, "kind {:?} s {}", kind, s);
        }
    }
}
