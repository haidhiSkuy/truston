use reqwest::Client;
use std::time::Duration;
use async_trait::async_trait;
use crate::utils::errors::TrustonError;

#[async_trait]
pub trait TritonClient: Send + Sync {
    async fn is_server_live(&self) -> Result<bool, TrustonError>;
}

pub struct TritonRestClient {
    pub base_url: String,
    pub http: Client,
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
            .map_err(|e| TrustonError::Http(format!("Request failed to send: {}", e)))?;
        
        tracing::info!("is_server_live: {} -> {}", url, resp.status());

        let status = resp.status();
        if status.is_success() {
            Ok(true) 
        } else {
            let status_code = status.as_u16();
            let body_text = resp.text().await.unwrap_or_else(|_| "No response body".to_string());
            let error_message = format!("Server is dead or unhealthy. Status: {}. Response body: {}", status_code, body_text);
            Err(TrustonError::HttpErrorResponse(status_code, error_message))
        }
    }
}

// ############################ UNIT TEST ################################
#[cfg(test)]
mod tests {
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
    async fn server_unreachable() {
        let client = TritonRestClient::new("http://localhost:12345");
        let result = client.is_server_live().await;
        assert!(matches!(result, Err(TrustonError::Http(_))));
    }
}
