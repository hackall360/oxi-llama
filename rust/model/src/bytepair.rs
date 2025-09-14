use crate::{
    text_processor::TextProcessor,
    vocabulary::{Special, Vocabulary, TOKEN_TYPE_CONTROL, TOKEN_TYPE_NORMAL},
};
use fancy_regex::Regex;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder, models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel, AddedToken, Tokenizer,
};

const DEFAULT_PRE_REGEX: &str =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

pub struct BytePairEncoding {
    tokenizer: Tokenizer,
    vocab: Vocabulary,
    regex: Regex,
}

impl BytePairEncoding {
    pub fn from_file(json: &str, vocab: Vocabulary) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let regex = Regex::new(DEFAULT_PRE_REGEX)?;
        Ok(Self {
            tokenizer,
            vocab,
            regex,
        })
    }

    pub fn from_vocab_files(encoder: &str, merges: &str) -> anyhow::Result<Self> {
        let file = File::open(encoder)?;
        let mut map: HashMap<String, u32> = serde_json::from_reader(file)?;
        let mut values = vec![String::new(); map.len()];
        for (tok, id) in &map {
            if (*id as usize) >= values.len() {
                values.resize(*id as usize + 1, String::new());
            }
            values[*id as usize] = tok.clone();
        }
        let mut types = vec![TOKEN_TYPE_NORMAL; values.len()];
        let mut next_id = values.len() as u32;
        for special in ["<|begin_of_text|>", "<|end_of_text|>"] {
            if !map.contains_key(special) {
                map.insert(special.to_string(), next_id);
                values.push(special.to_string());
                types.push(TOKEN_TYPE_CONTROL);
                next_id += 1;
            } else if let Some(&id) = map.get(special) {
                if (id as usize) >= types.len() {
                    types.resize(id as usize + 1, TOKEN_TYPE_NORMAL);
                }
                types[id as usize] = TOKEN_TYPE_CONTROL;
            }
        }

        let reader = BufReader::new(File::open(merges)?);
        let mut merges_vec = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            if let Some((a, b)) = line.split_once(' ') {
                merges_vec.push((a.to_string(), b.to_string()));
            }
        }

        let bpe = BPE::builder()
            .vocab_and_merges(map.clone(), merges_vec)
            .build()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut tokenizer = Tokenizer::new(bpe);
        let pre = ByteLevel::default().add_prefix_space(false);
        tokenizer.with_pre_tokenizer(pre);
        let dec = ByteLevelDecoder::default().add_prefix_space(false);
        tokenizer.with_decoder(dec);

        tokenizer.add_special_tokens(&[
            AddedToken::from("<|begin_of_text|>", true),
            AddedToken::from("<|end_of_text|>", true),
        ]);

        let mut vocab = Vocabulary::new();
        vocab.values = values;
        vocab.types = types;
        vocab.scores = vec![0.0; vocab.values.len()];
        if let Some(&id) = map.get("<|begin_of_text|>") {
            vocab.bos = vec![id as i32];
        }
        if let Some(&id) = map.get("<|end_of_text|>") {
            vocab.eos = vec![id as i32];
        }

        let regex = Regex::new(DEFAULT_PRE_REGEX)?;
        Ok(Self {
            tokenizer,
            vocab,
            regex,
        })
    }

    pub fn split(&self, s: &str) -> Vec<String> {
        let mut out = Vec::new();
        for m in self.regex.find_iter(s) {
            if let Ok(mat) = m {
                out.push(mat.as_str().to_string());
            }
        }
        out
    }
}

impl TextProcessor for BytePairEncoding {
    fn encode(&self, s: &str, add_special: bool) -> Result<Vec<i32>, anyhow::Error> {
        let enc = self
            .tokenizer
            .encode(s, add_special)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(enc.get_ids().iter().map(|&id| id as i32).collect())
    }

    fn decode(&self, ids: &[i32]) -> Result<String, anyhow::Error> {
        let ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }

    fn is(&self, id: i32, special: Special) -> bool {
        self.vocab.is(id, special)
    }

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }
}
