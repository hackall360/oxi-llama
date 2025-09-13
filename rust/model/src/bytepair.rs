use crate::{
    text_processor::TextProcessor,
    vocabulary::{Special, Vocabulary},
};
use tokenizers::Tokenizer;

pub struct BytePairEncoding {
    tokenizer: Tokenizer,
    vocab: Vocabulary,
}

impl BytePairEncoding {
    pub fn from_file(json: &str, vocab: Vocabulary) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(json).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Self { tokenizer, vocab })
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
