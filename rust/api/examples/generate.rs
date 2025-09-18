use api::{Client, GenerateRequest, GenerateResponse};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;
    let req = GenerateRequest {
        model: "gemma2".into(),
        prompt: "how many planets are there?".into(),
        stream: Some(false),
        ..Default::default()
    };
    client
        .generate(&req, |resp: GenerateResponse| {
            println!("{}", resp.response);
            Ok(())
        })
        .await?;
    Ok(())
}
