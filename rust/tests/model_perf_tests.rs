use anyhow::Result;

#[path = "common/mod.rs"]
mod common;

use common::{build_generate_request, IntegrationTest, ModelStatus};

const SHAKESPEARE: &str = include_str!("../../integration/testdata/shakespeare.txt");

#[tokio::test(flavor = "multi_thread")]
async fn test_long_prompt_metrics() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let excerpt = &SHAKESPEARE[..SHAKESPEARE.len().min(1024)];
    let req = build_generate_request(&format!("summarize the following: {excerpt}"));
    let outcome = ctx.generate(req).await?;
    let final_resp = outcome.final_response().expect("final response");

    assert!(
        final_resp
            .metrics
            .eval_count
            .map_or(false, |count| count > 0),
        "expected non-zero eval count"
    );
    assert!(
        final_resp
            .metrics
            .eval_duration
            .map_or(false, |duration| duration > 0),
        "expected eval duration to be populated"
    );

    Ok(())
}
