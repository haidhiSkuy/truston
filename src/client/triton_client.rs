//! Triton Inference Server client implementations.
//!
//! This module provides the main client interface for communicating with
//! Triton Inference Server via its REST API.


use reqwest::Client;
use std::time::Duration;
use async_trait::async_trait;
use crate::utils::errors::TrustonError;
use crate::client::io::{
    DataType, 
    InferInput, 
    InferInputPayload,
    TritonServerResponse,
    InferRequest,
    InferResponse,
    InferResults, 
    InferOutput,
};
use num_traits::NumCast;
use serde_json;

/// Trait defining the core operations for a Triton Inference Server client.
///
/// This trait can be implemented for different communication protocols
/// (REST, gRPC, etc.). Currently, only REST is implemented via `TritonRestClient`.
#[async_trait]
pub trait TritonClient: Send + Sync {
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

        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| TrustonError::Http(e))?;

        tracing::info!("is_server_live: {} -> {}", url, resp.status());

        let status = resp.status();
        if status.is_success() {
            Ok(true)
        } else {
            let status_code = status.as_u16();
            let body_text = resp
                .text()
                .await
                .unwrap_or_else(|_| "No response body".to_string());
            let error_message = format!(
                "Server is dead or unhealthy. Status: {}. Response body: {}",
                status_code, body_text
            );
            Err(TrustonError::ServerError{status: status_code, message: error_message})
        }
    }
}

impl TritonRestClient {

    /// Converts an `InferInput` into the JSON payload format required by Triton.
    ///
    /// This is an internal method that handles the conversion of Rust types
    /// to Triton's JSON format.
    /// 
    fn convert_input<'a>(
        &self,
        infer_input: &'a InferInput,
    ) -> InferInputPayload<'a, serde_json::Value> {
        let (datatype, data_json) = match &infer_input.input_data {
            DataType::Bool(v) => ("BOOL", serde_json::json!(v)),
            DataType::U8(v) => ("UINT8", serde_json::json!(v)),
            DataType::U16(v) => ("UINT16", serde_json::json!(v)),
            DataType::U64(v) => ("UINT64", serde_json::json!(v)),
            DataType::I8(v) => ("INT8", serde_json::json!(v)),
            DataType::I16(v) => ("INT16", serde_json::json!(v)),
            DataType::I32(v) => ("INT32", serde_json::json!(v)),
            DataType::I64(v) => ("INT64", serde_json::json!(v)),
            DataType::F32(v) => ("FP32", serde_json::json!(v)),
            DataType::F64(v) => ("FP64", serde_json::json!(v)),
            DataType::String(v) => ("STRING", serde_json::json!(v)),
            DataType::Bf16(v) => ("BF16", serde_json::json!(v)),
            DataType::Raw(v) => ("none", serde_json::json!(v)),
        };

        InferInputPayload {
            name: &infer_input.input_name,
            shape: infer_input.input_shape.clone(),
            datatype,
            data: data_json,
        }
    }

    /// Converts Triton's JSON response data to a typed Rust vector.
    ///
    /// This is an internal method that uses `NumCast` for safe numeric conversions.
    fn convert_output<T: NumCast>(&self, output_data: &TritonServerResponse) -> Option<Vec<T>> {
        match output_data.datatype.as_str() {
            "FP32" | "FP64" => output_data.data.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_f64())
                    .filter_map(|num| NumCast::from(num))
                    .collect()
            }),
            "UINT8" | "UINT16" | "UINT32" | "UINT64" => output_data.data.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_u64())
                    .filter_map(|num| NumCast::from(num))
                    .collect()
            }),
            "INT8" | "INT16" | "INT32" | "INT64" => output_data.data.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_i64())
                    .filter_map(|num| NumCast::from(num))
                    .collect()
            }),
            "BOOL" => output_data.data.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_bool())
                    .filter_map(|b| NumCast::from(b as u8))
                    .collect()
            }),
            _ => None,
        }
    }
    fn convert_output_string(&self, output_data: &TritonServerResponse) -> Option<Vec<String>> {
        match output_data.datatype.as_str() {
            "STRING" => output_data.data.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|item| item.as_str().map(|s| s.to_string()))
                    .collect()
            }),
            _ => None,
        }
    }


    pub async fn infer(
        &self,
        inputs: Vec<InferInput>,
        model_name: &str,
    ) -> Result<InferResults, TrustonError> {
        let url = format!("{}/v2/models/{}/infer", self.base_url, model_name);

        let input_payloads: Vec<_> = inputs.iter().map(|inp| self.convert_input(inp)).collect();

        let request = InferRequest {
            inputs: input_payloads,
        };

        let resp = self.http.post(&url).json(&request).send().await?;

        let status = resp.status();

        if !status.is_success() {
            let error_body = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error body".to_string());
            return Err(TrustonError::InferenceError(error_body));
        }

        let response_struct: InferResponse = resp
            .json::<InferResponse>()
            .await
            .map_err(|e| TrustonError::ParseError(e.to_string()))?;

 
        let mut converted_outputs = Vec::new();
        for output in &response_struct.outputs {
            let data = match output.datatype.as_str() {
                "UINT8" => self.convert_output::<u8>(output).map(DataType::U8), 
                "UINT16" => self.convert_output::<u16>(output).map(DataType::U16),
                "UINT64" => self.convert_output::<u64>(output).map(DataType::U64),
                "INT8" => self.convert_output::<i8>(output).map(DataType::I8),
                "INT16" => self.convert_output::<i16>(output).map(DataType::I16),
                "INT32" => self.convert_output::<i32>(output).map(DataType::I32),
                "INT64" => self.convert_output::<i64>(output).map(DataType::I64),
                "FP32" => self.convert_output::<f32>(output).map(DataType::F32),
                "FP64" => self.convert_output::<f64>(output).map(DataType::F64),
                "BF16" => self.convert_output::<u16>(output).map(DataType::Bf16),
                "STRING" => self.convert_output_string(output).map(DataType::String), 
            
                _ => Some(DataType::Raw(output.data.clone())),
            };
        
            if let Some(data) = data {
                converted_outputs.push(InferOutput {
                    name: output.name.clone(),
                    datatype: output.datatype.clone(),
                    shape: output.shape.clone(),
                    data,
                });
            }
        }
        Ok(InferResults { outputs: converted_outputs })
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
