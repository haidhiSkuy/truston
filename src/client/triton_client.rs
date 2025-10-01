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

    /// Convert the output data from a Triton server response into a vector of numeric values.
    ///
    /// This function attempts to parse the raw JSON `data` field returned by the Triton Inference Server
    /// into a strongly-typed `Vec<T>`, where `T` implements [`NumCast`].  
    /// It supports multiple Triton output datatypes (`FP32`, `INT64`, `BOOL`, etc.)
    /// and automatically casts the values into the requested type.
    ///
    /// # Type Parameters
    /// * `T: NumCast` - The target numeric type (e.g., `f32`, `f64`, `i32`, `u64`, etc.).
    ///
    /// # Arguments
    /// * `output_data` - A reference to a [`TritonServerResponse`] containing the model output.
    ///
    /// # Supported Datatypes
    /// - Floating point: `"FP32"`, `"FP64"` → parsed as `f64` then cast into `T`.
    /// - Unsigned integers: `"UINT8"`, `"UINT16"`, `"UINT32"`, `"UINT64"` → parsed as `u64` then cast.
    /// - Signed integers: `"INT8"`, `"INT16"`, `"INT32"`, `"INT64"` → parsed as `i64` then cast.
    /// - Boolean: `"BOOL"` → parsed as `bool`, then converted to `0` or `1` (`u8`) before casting.
    /// - Anything else returns `None`.
    ///
    /// # Returns
    /// * `Some(Vec<T>)` if the datatype is supported and the cast succeeds.
    /// * `None` if the datatype is unsupported or the JSON field is invalid.
    ///
    /// # Behavior
    /// - Iterates over the JSON array inside `data`.
    /// - Uses `filter_map` twice:
    ///   1. To parse the JSON into a base type (`f64`, `i64`, `u64`, or `bool`).
    ///   2. To cast the parsed value into the target type `T`.
    /// - Invalid entries or failed casts are skipped silently.
    ///
    /// # Example
    /// ```ignore
    /// let response: TritonServerResponse = client.infer(...).await?;
    ///
    /// // Convert float output
    /// if let Some(values) = my_client.convert_output::<f32>(&response) {
    ///     println!("Model float output: {:?}", values);
    /// }
    ///
    /// // Convert integer output
    /// if let Some(values) = my_client.convert_output::<i64>(&response) {
    ///     println!("Model int output: {:?}", values);
    /// }
    ///
    /// // Convert boolean output
    /// if let Some(values) = my_client.convert_output::<u8>(&response) {
    ///     println!("Model bool output (as 0/1): {:?}", values);
    /// }
    /// ```
    ///
    /// # Notes
    /// - The function does not fail hard: if a single element in the array fails parsing/casting,
    ///   it is skipped, but the rest of the vector is still returned.
    /// - For non-numeric outputs like `"STRING"`, use [`convert_output_string`] instead.
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

    /// Convert the output data from a Triton server response into a vector of strings.
    ///
    /// # Arguments
    /// * `output_data` - A reference to a [`TritonServerResponse`] object that contains
    ///   the inference result returned by the Triton Inference Server.
    ///
    /// # Returns
    /// * `Some(Vec<String>)` if:
    ///   - The `datatype` of the output is `"STRING"`.
    ///   - The `data` field can be parsed as an array of string values.
    /// * `None` if the `datatype` is not `"STRING"` or the data is not an array of strings.
    ///
    /// # Behavior
    /// - When the datatype is `"STRING"`, this function attempts to parse the `data`
    ///   field as an array of JSON values and filter out only the valid string entries.
    /// - Non-string entries inside the array will be ignored (they are skipped using
    ///   `filter_map`).
    ///
    /// # Example
    /// ```ignore
    /// let response: TritonServerResponse = client.infer(...).await?;
    /// if let Some(strings) = my_client.convert_output_string(&response) {
    ///     println!("Model output: {:?}", strings);
    /// } else {
    ///     println!("No valid string output found.");
    /// }
    /// ```
    ///
    /// # Notes
    /// - This helper is only meaningful for Triton model outputs with `datatype = "STRING"`.
    /// - For numeric outputs (e.g., `"FP32"`, `"INT64"`), consider using a different
    ///   converter function.
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

    /// Perform an inference request to the Triton Inference Server.
    ///
    /// This method sends a `POST` request to the Triton server's
    /// `/v2/models/{model_name}/infer` endpoint with the provided input tensors,
    /// waits for the response, parses it into structured outputs, and converts the
    /// raw response data into strongly-typed [`InferResults`].
    ///
    /// # Arguments
    /// * `inputs` - A list of [`InferInput`] objects representing the input tensors
    ///   (name, datatype, shape, and values) that will be sent to the model.
    /// * `model_name` - The name of the deployed model to query on the Triton server.
    ///
    /// # Returns
    /// * `Ok(InferResults)` - On success, containing a vector of [`InferOutput`] entries.
    /// * `Err(TrustonError)` - On failure, with possible variants:
    ///   - [`TrustonError::InferenceError`] if the server returned a non-2xx response
    ///     (includes the error body if available).
    ///   - [`TrustonError::ParseError`] if the response could not be deserialized into [`InferResponse`].
    ///   - Any other error bubbled up from the HTTP client (e.g., connection failure).
    ///
    /// # Supported Datatypes
    /// The server response is parsed into [`DataType`] variants depending on `datatype`:
    /// - `"UINT8"`, `"UINT16"`, `"UINT64"` → parsed into [`DataType::U8`], [`DataType::U16`], [`DataType::U64`]
    /// - `"INT8"`, `"INT16"`, `"INT32"`, `"INT64"` → parsed into [`DataType::I8`], [`DataType::I16`], [`DataType::I32`], [`DataType::I64`]
    /// - `"FP32"`, `"FP64"` → parsed into [`DataType::F32`], [`DataType::F64`]
    /// - `"BF16"` → parsed as `u16` and wrapped in [`DataType::Bf16`]
    /// - `"STRING"` → parsed into [`DataType::String`]
    /// - Any unknown datatype → stored raw in [`DataType::Raw`] with the original JSON payload.
    ///
    /// # Example
    /// ```ignore
    /// let client = TritonRestClient::new("http://localhost:8000");
    ///
    /// let input = InferInput::new("input_tensor", vec![1, 16], DataType::F32, vec![0.1f32; 16]);
    ///
    /// match client.infer(vec![input], "my_model").await {
    ///     Ok(results) => {
    ///         for out in results.outputs {
    ///             println!("Output {}: {:?}", out.name, out.data);
    ///         }
    ///     }
    ///     Err(e) => eprintln!("Inference failed: {:?}", e),
    /// }
    /// ```
    ///
    /// # Notes
    /// - Automatically converts JSON output values into the appropriate Rust types.
    /// - If conversion fails for a particular value, it will be skipped silently.
    /// - Any datatype not explicitly supported will be returned as raw JSON via `DataType::Raw`.
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
