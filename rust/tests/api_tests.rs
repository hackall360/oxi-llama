use anyhow::Result;
use serde_json::json;

#[path = "common/mod.rs"]
mod common;

use common::{
    build_chat_request, build_generate_request, IntegrationTest, ModelStatus, BLUE_SKY_KEYWORDS,
};

fn contains_any(text: &str, expected: &[&str]) -> bool {
    let lower = text.to_lowercase();
    expected.iter().any(|needle| lower.contains(needle))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_api_generate() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let mut req = build_generate_request("why is the sky blue? be brief");
    req.options.insert("temperature".into(), json!(0));
    req.options.insert("seed".into(), json!(123));

    let outcome = ctx.generate(req).await?;
    assert!(
        !outcome.responses.is_empty(),
        "no streaming updates returned"
    );
    assert!(
        contains_any(&outcome.full_text, BLUE_SKY_KEYWORDS),
        "response did not mention sky details: {}",
        outcome.full_text
    );

    let final_resp = outcome.final_response().expect("final response");
    assert!(!final_resp.model.is_empty(), "model field missing");
    assert!(
        !final_resp.context.is_empty(),
        "context should not be empty"
    );
    assert!(
        final_resp.metrics.total_duration.is_some(),
        "missing total duration"
    );
    assert!(
        final_resp.metrics.load_duration.is_some(),
        "missing load duration"
    );
    assert!(
        final_resp.metrics.eval_duration.is_some(),
        "missing eval duration"
    );

    let running = ctx.list_running().await?;
    assert!(
        running
            .models
            .iter()
            .any(|m| m.name.contains(ctx.smol_model())),
        "model should be listed as running"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_api_chat() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let mut req = build_chat_request("why is the sky blue? be brief");
    req.options.insert("temperature".into(), json!(0));
    req.options.insert("seed".into(), json!(321));

    let outcome = ctx.chat(req).await?;
    assert!(!outcome.responses.is_empty(), "no chat responses received");
    assert!(
        contains_any(&outcome.transcript, BLUE_SKY_KEYWORDS),
        "chat response missing expected keywords: {}",
        outcome.transcript
    );

    let final_resp = outcome.responses.last().expect("final chat");
    assert!(!final_resp.model.is_empty(), "model field missing");
    assert!(
        final_resp.metrics.total_duration.is_some(),
        "missing total duration"
    );

    Ok(())
}
