use anyhow::Result;

#[path = "common/mod.rs"]
mod common;

use common::{build_generate_request, IntegrationTest, ModelStatus, BLUE_SKY_KEYWORDS};

fn contains_any(text: &str, expected: &[&str]) -> bool {
    let lower = text.to_lowercase();
    expected.iter().any(|needle| lower.contains(needle))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_blue_sky_generation() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let req = build_generate_request("why is the sky blue?");
    let outcome = ctx.generate(req).await?;
    assert!(
        contains_any(&outcome.full_text, BLUE_SKY_KEYWORDS),
        "model response missing expected keywords: {}",
        outcome.full_text
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unicode_generation() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let req = build_generate_request("Provide a joyful response with multiple emoji icons");
    let outcome = ctx.generate(req).await?;
    let has_emoji = ["😀", "😁", "😂", "😊", "😄", "😃"]
        .iter()
        .any(|emoji| outcome.full_text.contains(emoji));
    assert!(
        has_emoji,
        "expected emoji output, got: {}",
        outcome.full_text
    );

    Ok(())
}
