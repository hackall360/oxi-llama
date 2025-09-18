use api::{Client, ProgressResponse, PullRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::from_env()?;
    let req = PullRequest {
        model: "mistral".into(),
        ..Default::default()
    };
    client
        .pull(&req, |resp: ProgressResponse| {
            println!(
                "Progress: status={}, total={}, completed={}",
                resp.status, resp.total, resp.completed
            );
            Ok(())
        })
        .await?;
    Ok(())
}
