use anyhow::Result;

#[path = "common/mod.rs"]
mod common;

use api::Message;
use common::{
    build_chat_request, build_generate_request, IntegrationTest, ModelStatus, BLUE_SKY_KEYWORDS,
};

fn contains_any(text: &str, expected: &[&str]) -> bool {
    let lower = text.to_lowercase();
    expected.iter().any(|needle| lower.contains(needle))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_generate_with_history() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };
    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let first = ctx
        .generate(build_generate_request(
            "why is the sky blue? explain briefly",
        ))
        .await?;
    assert!(contains_any(&first.full_text, BLUE_SKY_KEYWORDS));

    let mut follow_up = build_generate_request("tell me more!");
    follow_up.context = Some(first.context.clone());
    let follow = ctx.generate(follow_up).await?;
    assert!(!follow.full_text.is_empty(), "follow up response was empty");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_chat_with_history() -> Result<()> {
    let Some(ctx) = IntegrationTest::new().await? else {
        return Ok(());
    };
    if matches!(
        ctx.ensure_model(ctx.smol_model()).await?,
        ModelStatus::Skipped
    ) {
        return Ok(());
    }

    let first_prompt = "why is the sky blue?";
    let first = ctx.chat(build_chat_request(first_prompt)).await?;
    let assistant = first
        .responses
        .last()
        .map(|resp| resp.message.clone())
        .unwrap_or(Message {
            role: "assistant".into(),
            content: String::new(),
            thinking: String::new(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_name: String::new(),
        });

    let mut follow = build_chat_request("tell me more about atmospheric scattering");
    follow.messages.insert(
        0,
        Message {
            role: "user".into(),
            content: first_prompt.to_string(),
            thinking: String::new(),
            images: Vec::new(),
            tool_calls: Vec::new(),
            tool_name: String::new(),
        },
    );
    follow.messages.insert(1, assistant);
    let second = ctx.chat(follow).await?;
    assert!(contains_any(&second.transcript, BLUE_SKY_KEYWORDS));

    Ok(())
}
