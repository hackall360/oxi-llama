use anyhow::Result;

#[path = "common/mod.rs"]
mod common;

use common::{IntegrationTest, ModelStatus};

#[tokio::test(flavor = "multi_thread")]
async fn test_quantization_level_exposed() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let info = ctx.show(ctx.smol_model()).await?;
    assert!(
        !info.details.quantization_level.is_empty(),
        "quantization level should be reported"
    );

    Ok(())
}
