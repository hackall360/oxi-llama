use crate::vocabulary::Vocabulary;

pub trait TextProcessor: Send + Sync {
    fn encode(&self, s: &str, add_special: bool) -> Result<Vec<i32>, anyhow::Error>;
    fn decode(&self, ids: &[i32]) -> Result<String, anyhow::Error>;
    fn is(&self, id: i32, special: crate::vocabulary::Special) -> bool;
    fn vocabulary(&self) -> &Vocabulary;
}
