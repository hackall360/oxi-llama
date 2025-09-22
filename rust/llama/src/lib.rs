use schema::json_schema_str_to_grammar;

/// Convert a JSON schema into a grammar representation used by llama.cpp.
///
/// Set `force_gbnf` to `true` to always emit a GBNF grammar even when the
/// llguidance format would be supported by the underlying runtime.
pub fn schema_to_grammar(schema: &str, force_gbnf: bool) -> Result<String, SchemaError> {
    json_schema_str_to_grammar(schema, force_gbnf)
}

/// Convenience wrapper that always emits a GBNF grammar and returns the
/// encoded bytes on success.
pub fn schema_to_grammar_safe(schema: &str) -> Option<Vec<u8>> {
    schema_to_grammar(schema, true).map(|g| g.into_bytes()).ok()
}

pub use schema::CommonGrammarBuilder;
pub use schema::CommonGrammarOptions;
pub use schema::SchemaError;
