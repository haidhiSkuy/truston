use reqwest::Client;
use std::time::Duration;
use async_trait::async_trait;
use crate::utils::errors::TrustonError;

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

// ############################ UNIT TEST ################################
#[cfg(test)]
mod tests {
    use core::panic;

    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_is_server_live() {
        crate::init_tracing();

        let client = TritonRestClient::new("http://localhost:50000");
        let result = client.is_server_live().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_is_server_not_live() {
        let client = TritonRestClient::new("http://localhost:12345");
        let result = client.is_server_live().await;
        assert!(matches!(result, Err(TrustonError::Http(_))));
    }
}