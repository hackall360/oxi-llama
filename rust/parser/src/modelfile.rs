use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use sha2::{Digest, Sha256};
use thiserror::Error;
use users::{get_current_uid, get_user_by_name, get_user_by_uid, os::unix::UserExt};
use walkdir::WalkDir;

use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    pub name: String,
    pub args: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Modelfile {
    pub commands: Vec<Command>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CreateRequest {
    pub from: String,
    pub files: HashMap<String, String>,
    pub adapters: HashMap<String, String>,
    pub template: String,
    pub license: Option<Value>,
    pub system: String,
    pub parameters: HashMap<String, Value>,
    pub messages: Vec<Message>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ParserError {
    #[error("no FROM line")]
    MissingFrom,
    #[error("message role must be one of \"system\", \"user\", or \"assistant\"")]
    InvalidMessageRole,
    #[error(
        "command must be one of \"from\", \"license\", \"template\", \"system\", \"adapter\", \"parameter\", or \"message\""
    )]
    InvalidCommand,
    #[error("unexpected end of file")]
    UnexpectedEOF,
}

fn is_valid_command(cmd: &str) -> bool {
    matches!(
        cmd.to_lowercase().as_str(),
        "from" | "license" | "template" | "system" | "adapter" | "parameter" | "message"
    )
}

fn is_valid_role(role: &str) -> bool {
    matches!(role, "system" | "user" | "assistant")
}

/// Parse a Modelfile from a reader
pub fn parse_file<R: Read>(r: R) -> Result<Modelfile> {
    let reader = BufReader::new(r);
    let mut lines = reader.lines().peekable();
    let mut commands = Vec::new();
    let mut seen_from = false;

    while let Some(line) = lines.next() {
        let mut line = line?;
        // remove carriage return for windows line endings
        if line.ends_with('\r') {
            line.pop();
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }

        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap();
        let rest = parts.next().unwrap_or("").trim_start();

        match cmd.to_lowercase().as_str() {
            "from" => {
                let arg = parse_arg(rest, &mut lines)?;
                commands.push(Command {
                    name: "model".into(),
                    args: arg.clone(),
                });
                seen_from = true;
            }
            "adapter" | "license" | "template" | "system" => {
                let arg = parse_arg(rest, &mut lines)?;
                commands.push(Command {
                    name: cmd.to_lowercase(),
                    args: arg,
                });
            }
            "message" => {
                let mut subparts = rest.splitn(2, char::is_whitespace);
                let role = subparts.next().unwrap_or("");
                if !is_valid_role(role) {
                    bail!(ParserError::InvalidMessageRole);
                }
                let msg_rest = subparts.next().unwrap_or("").trim_start();
                let msg = parse_arg(msg_rest, &mut lines)?;
                commands.push(Command {
                    name: "message".into(),
                    args: format!("{}: {}", role, msg),
                });
            }
            "parameter" => {
                let mut subparts = rest.splitn(2, char::is_whitespace);
                let name = subparts.next().unwrap_or("").to_string();
                let val_rest = subparts.next().unwrap_or("").trim_start();
                if val_rest.is_empty() {
                    return Err(ParserError::UnexpectedEOF.into());
                }
                let val = parse_arg(val_rest, &mut lines)?;
                commands.push(Command { name, args: val });
            }
            other => {
                if is_valid_command(other) {
                    bail!("unhandled command");
                }
                bail!(ParserError::InvalidCommand);
            }
        }
    }

    if !seen_from {
        bail!(ParserError::MissingFrom);
    }

    Ok(Modelfile { commands })
}

fn parse_arg<I>(first: &str, lines: &mut std::iter::Peekable<I>) -> Result<String>
where
    I: Iterator<Item = io::Result<String>>,
{
    if first.starts_with("\"\"\"") {
        let mut acc = String::new();
        let rest = &first[3..];
        if let Some(idx) = rest.find("\"\"\"") {
            acc.push_str(&rest[..idx]);
            return Ok(acc);
        }
        acc.push_str(rest);
        acc.push('\n');
        while let Some(line) = lines.next() {
            let line = line?;
            if let Some(idx) = line.find("\"\"\"") {
                acc.push_str(&line[..idx]);
                return Ok(acc);
            }
            acc.push_str(&line);
            acc.push('\n');
        }
        bail!(ParserError::UnexpectedEOF);
    } else if first.starts_with('"') {
        if let Some(idx) = first[1..].find('"') {
            return Ok(first[1..1 + idx].to_string());
        }
        bail!(ParserError::UnexpectedEOF);
    } else {
        Ok(first.trim_end().to_string())
    }
}

/// Expand a path relative to `relative_dir` resolving `~` and `~user` prefixes.
pub fn expand_path<P: AsRef<Path>, Q: AsRef<Path>>(path: P, relative_dir: Q) -> Result<PathBuf> {
    expand_path_impl(
        path.as_ref(),
        relative_dir.as_ref(),
        || {
            let uid = get_current_uid();
            get_user_by_uid(uid)
                .map(|u| PathBuf::from(u.home_dir()))
                .ok_or_else(|| anyhow::anyhow!("failed to get current user"))
        },
        |name| {
            get_user_by_name(name)
                .map(|u| PathBuf::from(u.home_dir()))
                .ok_or_else(|| anyhow::anyhow!("failed to find user"))
        },
    )
}

pub(crate) fn expand_path_impl<F, G>(
    path: &Path,
    relative_dir: &Path,
    current_home: F,
    lookup_home: G,
) -> Result<PathBuf>
where
    F: Fn() -> Result<PathBuf>,
    G: Fn(&str) -> Result<PathBuf>,
{
    let path_str = path.to_string_lossy().to_string();
    let result = if path.is_absolute() {
        PathBuf::from(&path_str)
    } else if path_str.starts_with('~') {
        let rest = &path_str[1..];
        let sep_idx = rest.find(|c| c == '/' || c == '\\');
        let (user_part, remaining) = match sep_idx {
            Some(i) => (&rest[..i], &rest[i + 1..]),
            None => (rest, ""),
        };
        if user_part.is_empty() {
            let home = current_home()?;
            Path::new(&home).join(remaining)
        } else {
            let home = lookup_home(user_part)?;
            Path::new(&home).join(remaining)
        }
    } else {
        let base = if relative_dir.as_os_str().is_empty() {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        } else if relative_dir.is_absolute() {
            relative_dir.to_path_buf()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(relative_dir)
        };
        base.join(path)
    };
    Ok(fs::canonicalize(&result).unwrap_or(result))
}

fn digest_for_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let _ = io::copy(&mut f, &mut hasher)?;
    let digest = hasher.finalize();
    Ok(format!("sha256:{:x}", digest))
}

fn file_digest_map(path: &Path) -> Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    let meta = fs::metadata(path)?;
    if meta.is_dir() {
        for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
            if entry.file_type().is_file() {
                let p = entry.path().to_path_buf();
                let digest = digest_for_file(&p)?;
                map.insert(p.to_string_lossy().to_string(), digest);
            }
        }
    } else {
        map.insert(path.to_string_lossy().to_string(), digest_for_file(path)?);
    }
    Ok(map)
}

impl Modelfile {
    pub fn create_request(&self, relative_dir: &Path) -> Result<CreateRequest> {
        let mut req = CreateRequest::default();
        let mut messages = Vec::new();
        let mut licenses = Vec::new();
        let mut params: HashMap<String, Value> = HashMap::new();

        for c in &self.commands {
            match c.name.as_str() {
                "model" => {
                    let p = expand_path(&c.args, relative_dir)?;
                    match file_digest_map(&p) {
                        Ok(m) => {
                            if req.files.is_empty() {
                                req.files = m;
                            } else {
                                req.files.extend(m);
                            }
                        }
                        Err(e) => {
                            if let Some(ioe) = e.downcast_ref::<io::Error>() {
                                if ioe.kind() == io::ErrorKind::NotFound {
                                    req.from = c.args.clone();
                                    continue;
                                }
                            }
                            return Err(e);
                        }
                    }
                }
                "adapter" => {
                    let p = expand_path(&c.args, relative_dir)?;
                    let m = file_digest_map(&p)?;
                    req.adapters = m;
                }
                "template" => req.template = c.args.clone(),
                "system" => req.system = c.args.clone(),
                "license" => licenses.push(c.args.clone()),
                "message" => {
                    if let Some((role, content)) = c.args.split_once(':') {
                        messages.push(Message {
                            role: role.trim().into(),
                            content: content.trim().into(),
                        });
                    }
                }
                other => {
                    let val = parse_param_value(&c.args);
                    if let Some(existing) = params.get_mut(other) {
                        match existing {
                            Value::Array(arr) => arr.push(val),
                            v => *v = Value::Array(vec![v.clone(), val]),
                        }
                    } else {
                        params.insert(other.to_string(), val);
                    }
                }
            }
        }
        if !messages.is_empty() {
            req.messages = messages;
        }
        if !licenses.is_empty() {
            req.license = Some(Value::Array(
                licenses.into_iter().map(Value::String).collect(),
            ));
        }
        if !params.is_empty() {
            req.parameters = params;
        }
        Ok(req)
    }
}

fn parse_param_value(s: &str) -> Value {
    if let Ok(i) = s.parse::<i64>() {
        Value::from(i)
    } else if let Ok(f) = s.parse::<f64>() {
        Value::from(f)
    } else {
        Value::String(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    #[test]
    fn parse_basic() {
        let input = "FROM model\nPARAMETER foo bar\n";
        let m = parse_file(Cursor::new(input)).unwrap();
        assert_eq!(m.commands.len(), 2);
    }

    #[test]
    fn expand_path_basic() {
        let cwd = std::env::current_dir().unwrap();
        let p = expand_path("./test", &cwd).unwrap();
        assert!(p.to_string_lossy().contains("test"));
    }

    #[test]
    fn expand_path_tilde() {
        let mock_current = || Ok(PathBuf::from("/home/testuser"));
        let mock_lookup = |name: &str| -> Result<PathBuf> {
            if name == "another" {
                Ok(PathBuf::from("/home/another"))
            } else {
                Err(anyhow::anyhow!("user not found"))
            }
        };
        let p = super::expand_path_impl(
            Path::new("~another/docs"),
            Path::new(""),
            mock_current,
            mock_lookup,
        )
        .unwrap();
        assert_eq!(p, PathBuf::from("/home/another/docs"));
    }

    #[test]
    fn expand_path_cases() {
        if cfg!(windows) {
            return;
        }
        let pwd = std::env::current_dir().unwrap();
        let cases = vec![
            ("~", "", PathBuf::from("/home/testuser"), false),
            (
                "~/myfolder/myfile.txt",
                "",
                PathBuf::from("/home/testuser/myfolder/myfile.txt"),
                false,
            ),
            (
                "~anotheruser/docs/file.txt",
                "",
                PathBuf::from("/home/anotheruser/docs/file.txt"),
                false,
            ),
            ("~nonexistentuser/file.txt", "", PathBuf::new(), true),
            (
                "relative/path/to/file",
                "",
                pwd.join("relative/path/to/file"),
                false,
            ),
            (
                "/absolute/path/to/file",
                "",
                PathBuf::from("/absolute/path/to/file"),
                false,
            ),
            (
                "/absolute/path/to/file",
                "someotherdir/",
                PathBuf::from("/absolute/path/to/file"),
                false,
            ),
            (".", pwd.to_str().unwrap(), pwd.clone(), false),
            (".", "", pwd.clone(), false),
            (
                "somefile",
                "somedir",
                pwd.join("somedir").join("somefile"),
                false,
            ),
        ];
        for (path, rel, expected, should_err) in cases {
            let res = super::expand_path_impl(
                Path::new(path),
                Path::new(rel),
                || Ok(PathBuf::from("/home/testuser")),
                |name| match name {
                    "testuser" => Ok(PathBuf::from("/home/testuser")),
                    "anotheruser" => Ok(PathBuf::from("/home/anotheruser")),
                    _ => Err(anyhow::anyhow!("user not found")),
                },
            );
            if should_err {
                assert!(res.is_err());
            } else {
                assert_eq!(res.unwrap(), expected);
            }
        }
    }
}
