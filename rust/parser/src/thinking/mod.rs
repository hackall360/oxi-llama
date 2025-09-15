pub mod parser;
pub mod template;

pub use parser::{Parser, State};
pub use template::infer_tags;
