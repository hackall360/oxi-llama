use api::{Client, GenerateRequest, GenerateResponse};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;
    let req = GenerateRequest {
        model: "gemma2".into(),
        prompt: "how many planets are there?".into(),
        ..Default::default()
    };
    client
        .generate(&req, |resp: GenerateResponse| {
            print!("{}", resp.response);
            Ok(())
        })
        .await?;
    println!();
    Ok(())
}
