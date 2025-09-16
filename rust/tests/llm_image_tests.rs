use anyhow::Result;
use serde_json::json;

#[path = "common/mod.rs"]
mod common;

use common::{build_generate_request, IntegrationTest, ModelStatus};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires optional vision models"]
async fn test_vision_model_reading_text() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    let model = "llama3.2-vision:latest";
    if matches!(ctx.ensure_model(model).await?, ModelStatus::Skipped) {
        return Ok(());
    }

    let mut req = build_generate_request("describe the text in this image");
    req.model = model.into();
    req.images.push(common::sample_image());
    req.options.insert("temperature".into(), json!(0));
    req.options.insert("seed".into(), json!(42));

    let outcome = ctx.generate(req).await?;
    assert!(
        !outcome.full_text.is_empty(),
        "vision model produced empty response"
    );

    Ok(())
}
