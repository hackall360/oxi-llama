use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

/// Representation of a tokenizer parsed from model files.
#[derive(Debug, PartialEq)]
pub struct Tokenizer {
    pub vocabulary: Vocabulary,
    pub special_vocabulary: Vec<SpecialVocabulary>,
    pub merges: Vec<String>,
    pub pre: String,
    pub template: String,
}

#[derive(Debug, PartialEq)]
pub struct Vocabulary {
    pub model: String,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub types: Vec<i32>,
}

#[derive(Debug, PartialEq)]
pub struct SpecialVocabulary {
    pub r#type: String,
    pub id: i32,
    pub content: String,
    pub add_token: bool,
    pub ids: Vec<i32>,
}

#[derive(Deserialize, Default)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    model: Option<ModelSection>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: i32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct ModelSection {
    #[serde(default)]
    vocab: BTreeMap<String, i32>,
}

struct Token {
    id: i32,
    content: String,
    special: bool,
    user_defined: bool,
}

/// Parse tokenizer information from a directory containing tokenizer files.
pub fn parse_tokenizer<P: AsRef<Path>>(dir: P, _special_token_types: &[&str]) -> Result<Tokenizer> {
    let dir = dir.as_ref();
    let tok_path = dir.join("tokenizer.json");
    let tj: TokenizerJson = if tok_path.exists() {
        let data = fs::read_to_string(&tok_path).context("reading tokenizer.json")?;
        serde_json::from_str(&data).context("parsing tokenizer.json")?
    } else {
        TokenizerJson::default()
    };

    let mut tokens: BTreeMap<i32, Token> = BTreeMap::new();
    if let Some(model) = tj.model {
        for (content, id) in model.vocab {
            tokens.insert(
                id,
                Token {
                    id,
                    content,
                    special: false,
                    user_defined: false,
                },
            );
        }
    }
    for t in tj.added_tokens {
        tokens.insert(
            t.id,
            Token {
                id: t.id,
                content: t.content,
                special: t.special,
                user_defined: true,
            },
        );
    }

    let mut vocab_tokens = Vec::new();
    let mut scores = Vec::new();
    let mut types = Vec::new();
    for token in tokens.values() {
        vocab_tokens.push(token.content.clone());
        scores.push(token.id as f32);
        let ty = if token.special {
            3
        } else if token.user_defined {
            4
        } else {
            1
        };
        types.push(ty);
    }
    let vocabulary = Vocabulary {
        model: "gpt2".into(),
        tokens: vocab_tokens,
        scores,
        types,
    };

    // tokenizer_config.json may contain chat_template
    let mut template = String::new();
    let cfg_path = dir.join("tokenizer_config.json");
    if cfg_path.exists() {
        let data = fs::read_to_string(&cfg_path).context("reading tokenizer_config.json")?;
        let v: Value = serde_json::from_str(&data).context("parsing tokenizer_config.json")?;
        if let Some(ct) = v.get("chat_template") {
            if let Some(s) = ct.as_str() {
                template = s.to_string();
            } else if let Some(arr) = ct.as_array() {
                for entry in arr {
                    if entry.get("name").and_then(|v| v.as_str()) == Some("default") {
                        if let Some(tpl) = entry.get("template").and_then(|v| v.as_str()) {
                            template = tpl.to_string();
                            break;
                        }
                    }
                }
            }
        }
    }

    Ok(Tokenizer {
        vocabulary,
        special_vocabulary: Vec::new(),
        merges: Vec::new(),
        pre: "default".into(),
        template,
    })
}
