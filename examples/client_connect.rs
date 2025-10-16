use tokio;
use truston::client::http::TritonRestClient;
use truston::init_tracing;

#[tokio::main]
async fn main() {
    init_tracing();

    let my_client = TritonRestClient::new("http://localhost:50000");
    let is_alive = my_client.is_server_live().await;
    match is_alive {
        Ok(_) => println!("Server is live!"),
        Err(e) => println!("error: {:#?}", e),
    }
}