use std::env;
use std::fs;

use api::{Client, GenerateRequest, GenerateResponse};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).expect("usage: <image name>");
    let img_data = fs::read(path)?;
    let client = Client::from_env()?;
    let req = GenerateRequest { model: "llava".into(), prompt: "describe this image".into(), images: vec![img_data], ..Default::default() };
    client.generate(&req, |resp: GenerateResponse| {
        print!("{}", resp.response);
        Ok(())
    }).await?;
    println!();
    Ok(())
}
