use api::openai::{ChatCompletionRequest, Message};
use api::Client;
use reqwest::Method;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client pointing at the local Ollama server
    let client = Client::from_env()?;

    // Build an OpenAI compatible chat completion request
    let req = ChatCompletionRequest {
        model: "llama3.1".into(),
        messages: vec![Message {
            role: "user".into(),
            content: serde_json::Value::String("Hello".into()),
            ..Default::default()
        }],
        ..Default::default()
    };

    // Send request to the OpenAI compatible endpoint
    let resp: serde_json::Value = client
        .do_request(Method::POST, "/v1/chat/completions", Some(&req))
        .await?;

    println!("{resp}");
    Ok(())
}
