use anyhow::Result;

#[path = "common/mod.rs"]
mod common;

use common::{IntegrationTest, ModelStatus};

#[tokio::test(flavor = "multi_thread")]
async fn test_show_reports_architecture_details() -> Result<()> {
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
    let arch_key = "general.architecture";
    let Some(arch_value) = info.model_info.get(arch_key).and_then(|v| v.as_str()) else {
        anyhow::bail!("show response missing {arch_key}");
    };
    assert!(
        !arch_value.is_empty(),
        "architecture string should not be empty"
    );

    let context_key = format!("{arch_value}.context_length");
    if let Some(len) = info.model_info.get(&context_key).and_then(|v| v.as_f64()) {
        assert!(len > 0.0, "context length should be positive");
    }

    Ok(())
}
