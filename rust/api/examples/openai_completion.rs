use api::openai::CompletionRequest;
use api::Client;
use reqwest::Method;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;
    let req = CompletionRequest {
        model: "llama3.1".into(),
        prompt: "Hello".into(),
        ..Default::default()
    };
    let resp: serde_json::Value = client
        .do_request(Method::POST, "/v1/completions", Some(&req))
        .await?;
    println!("{resp}");
    Ok(())
}
