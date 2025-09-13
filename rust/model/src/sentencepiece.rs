use crate::{
    text_processor::TextProcessor,
    vocabulary::{Special, Vocabulary, TOKEN_TYPE_BYTE},
};
use sentencepiece::SentencePieceProcessor;

pub struct SentencePieceModel {
    processor: Option<SentencePieceProcessor>,
    vocab: Vocabulary,
}

impl SentencePieceModel {
    pub fn from_file(model: &str, vocab: Vocabulary) -> anyhow::Result<Self> {
        let proc = SentencePieceProcessor::open(model)?;
        Ok(Self {
            processor: Some(proc),
            vocab,
        })
    }

    pub fn new(vocab: Vocabulary) -> Self {
        Self {
            processor: None,
            vocab,
        }
    }
}

impl TextProcessor for SentencePieceModel {
    fn encode(&self, s: &str, add_special: bool) -> Result<Vec<i32>, anyhow::Error> {
        if let Some(proc) = &self.processor {
            // If the entire string matches a known piece (e.g. special token),
            // use its id directly.
            let mut ids = if let Ok(Some(id)) = proc.piece_to_id(s) {
                vec![id as i32]
            } else {
                let pieces = proc.encode(s)?;
                pieces.iter().map(|p| p.id as i32).collect()
            };
            if add_special {
                ids = self.vocab.add_specials(ids);
            }
            Ok(ids)
        } else {
            Err(anyhow::anyhow!("encode not supported"))
        }
    }

    fn decode(&self, ids: &[i32]) -> Result<String, anyhow::Error> {
        if let Some(proc) = &self.processor {
            let ids32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
            return Ok(proc.decode_piece_ids(&ids32)?);
        }
        let mut out = String::new();
        let mut bytes = Vec::new();
        for &id in ids {
            if self.vocab.types[id as usize] == TOKEN_TYPE_BYTE {
                let token = &self.vocab.values[id as usize];
                if let Some(hex) = token.trim_start_matches("<0x").strip_suffix('>') {
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        bytes.push(b);
                        continue;
                    }
                }
            }
            if !bytes.is_empty() {
                match String::from_utf8(bytes.clone()) {
                    Ok(s) => out.push_str(&s),
                    Err(_) => {
                        for b in &bytes {
                            out.push(*b as char);
                        }
                    }
                }
                bytes.clear();
            }
            out.push_str(&self.vocab.values[id as usize]);
        }
        if !bytes.is_empty() {
            match String::from_utf8(bytes.clone()) {
                Ok(s) => out.push_str(&s),
                Err(_) => {
                    for b in &bytes {
                        out.push(*b as char);
                    }
                }
            }
        }
        Ok(out)
    }

    fn is(&self, id: i32, special: Special) -> bool {
        self.vocab.is(id, special)
    }

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }
}
