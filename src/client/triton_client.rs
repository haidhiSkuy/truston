use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

#[derive(Debug)]
pub enum TrustonError {
    Http(String),
}

#[async_trait]
pub trait TritonClient {
    async fn is_server_live(&self) -> Result<bool, TrustonError>;
}

pub struct TritonRestClient {
    base_url: String,
    http: Client,
}

impl TritonRestClient {
    pub fn new(base_url: &str) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("failed to build client");

        Self {
            base_url: base_url.to_string(),
            http,
        }
    }
}

#[async_trait]
impl TritonClient for TritonRestClient {
    async fn is_server_live(&self) -> Result<bool, TrustonError> {
        let url = format!("{}/v2/health/ready", self.base_url);

        let resp = self.http
            .get(&url)
            .send()
            .await
            .map_err(|e| TrustonError::Http(e.to_string()))?;
        
        tracing::info!("is_server_live: {} -> {}", url, resp.status());

        Ok(resp.status().is_success())
    }
}