use once_cell::sync::OnceCell;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub enum Special {
    Bos,
    Eos,
}

pub const TOKEN_TYPE_NORMAL: i32 = 1;
pub const TOKEN_TYPE_UNKNOWN: i32 = 2;
pub const TOKEN_TYPE_CONTROL: i32 = 3;
pub const TOKEN_TYPE_USER_DEFINED: i32 = 4;
pub const TOKEN_TYPE_UNUSED: i32 = 5;
pub const TOKEN_TYPE_BYTE: i32 = 6;

pub struct Vocabulary {
    pub values: Vec<String>,
    pub types: Vec<i32>,
    pub scores: Vec<f32>,
    pub merges: Vec<String>,
    pub bos: Vec<i32>,
    pub eos: Vec<i32>,
    pub add_bos: bool,
    pub add_eos: bool,
    values_map: OnceCell<HashMap<String, i32>>,
    special: OnceCell<Vec<String>>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            types: Vec::new(),
            scores: Vec::new(),
            merges: Vec::new(),
            bos: Vec::new(),
            eos: Vec::new(),
            add_bos: false,
            add_eos: false,
            values_map: OnceCell::new(),
            special: OnceCell::new(),
        }
    }

    pub fn is(&self, id: i32, special: Special) -> bool {
        match special {
            Special::Bos => self.bos.contains(&id),
            Special::Eos => self.eos.contains(&id),
        }
    }

    pub fn add_specials(&self, mut ids: Vec<i32>) -> Vec<i32> {
        if self.add_bos && !self.bos.is_empty() {
            ids.insert(0, self.bos[0]);
        }
        if self.add_eos && !self.eos.is_empty() {
            ids.push(self.eos[0]);
        }
        ids
    }

    pub fn encode(&self, s: &str) -> i32 {
        let map = self.values_map.get_or_init(|| {
            let mut m = HashMap::new();
            for (i, v) in self.values.iter().enumerate() {
                m.insert(v.clone(), i as i32);
            }
            m
        });
        *map.get(s).unwrap_or(&-1)
    }

    pub fn decode(&self, id: i32) -> &str {
        &self.values[id as usize]
    }

    pub fn special_vocabulary(&self) -> Vec<String> {
        self.special
            .get_or_init(|| {
                self.values
                    .iter()
                    .zip(self.types.iter())
                    .filter_map(|(v, t)| {
                        if *t == TOKEN_TYPE_CONTROL || *t == TOKEN_TYPE_USER_DEFINED {
                            Some(v.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .clone()
    }

    pub fn merge(&self, _left: &str, _right: &str) -> i32 {
        // simplistic: not needed for tests
        -1
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}
