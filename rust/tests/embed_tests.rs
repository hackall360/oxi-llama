use anyhow::Result;
use serde_json::json;

#[path = "common/mod.rs"]
mod common;

use api::{Duration as ApiDuration, EmbeddingRequest};
use common::{IntegrationTest, ModelStatus};

#[tokio::test(flavor = "multi_thread")]
async fn test_text_embeddings() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    let model = "all-minilm";
    if matches!(ctx.ensure_model(model).await?, ModelStatus::Skipped) {
        return Ok(());
    }

    let mut req = EmbeddingRequest::default();
    req.model = model.into();
    req.prompt = "why is the sky blue?".into();
    req.keep_alive = Some(ApiDuration(common::DEFAULT_GENERATE_TIMEOUT));
    req.options.insert("seed".into(), json!(123));

    let resp = ctx.embeddings(req).await?;
    assert!(
        !resp.embedding.is_empty(),
        "embedding vector should not be empty"
    );

    let magnitude: f64 = resp
        .embedding
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        magnitude.is_finite(),
        "embedding magnitude should be finite"
    );

    Ok(())
}
