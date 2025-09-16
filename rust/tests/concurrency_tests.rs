use anyhow::Result;
use tokio::task::JoinSet;

#[path = "common/mod.rs"]
mod common;

use common::{build_generate_request, IntegrationTest, ModelStatus};

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_generate() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };

    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let prompts = vec![
        (
            "why is the ocean blue? explain succinctly",
            vec!["rayleigh", "scatter", "water"],
        ),
        (
            "how do rainbows form?",
            vec!["prism", "refraction", "light"],
        ),
        (
            "summarize the composition of air",
            vec!["nitrogen", "oxygen", "carbon"],
        ),
    ];

    let mut jobs = JoinSet::new();
    for (prompt, keywords) in prompts {
        let ctx_clone = ctx.clone();
        let req = build_generate_request(prompt);
        jobs.spawn(async move {
            let outcome = ctx_clone.generate(req).await?;
            let output = outcome.full_text.to_lowercase();
            let success = keywords.iter().any(|kw| output.contains(kw));
            anyhow::ensure!(success, "response missing expected keywords: {}", output);
            Ok::<(), anyhow::Error>(())
        });
    }

    while let Some(result) = jobs.join_next().await {
        result??;
    }

    Ok(())
}
