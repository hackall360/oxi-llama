use api::{Client, ChatRequest, Message, ChatResponse};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;
    let messages = vec![
        Message { role: "system".into(), content: "Provide very brief, concise responses".into(), ..Default::default() },
        Message { role: "user".into(), content: "Name some unusual animals".into(), ..Default::default() },
        Message { role: "assistant".into(), content: "Monotreme, platypus, echidna".into(), ..Default::default() },
        Message { role: "user".into(), content: "which of these is the most dangerous?".into(), ..Default::default() },
    ];
    let req = ChatRequest { model: "llama3.2".into(), messages, ..Default::default() };
    client.chat(&req, |resp: ChatResponse| {
        print!("{}", resp.message.content);
        Ok(())
    }).await?;
    Ok(())
}
