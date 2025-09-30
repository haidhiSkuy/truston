use crate::client::triton_client::TritonRestClient;
use crate::models::input_model::{InferInput, InferData}; 
use crate::utils::errors::TrustonError; 
use serde::{Serialize, Deserialize};
use serde_json;

// Input
#[derive(Serialize)]
struct InferRequest<'a, T> {
    inputs: Vec<InferInputPayload<'a, T>>
}

#[derive(Serialize)]
struct InferInputPayload<'a, T> {
    name: &'a str,
    shape: Vec<usize>,
    datatype: &'a str,
    data: T,
}

// Output
#[derive(Debug, Deserialize)]
pub struct InferOutputData {
    pub name: String,
    pub shape: Vec<usize>,
    pub datatype: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct InferResponse {
    pub outputs: Vec<InferOutputData>, 
}


impl TritonRestClient {
    fn convert_input<'a>(&self, infer_input: &'a InferInput) -> InferInputPayload<'a, serde_json::Value> {
        let (datatype, data_json) = match &infer_input.input_data {
            InferData::Bool(v)   => ("BOOL", serde_json::json!(v)),
            InferData::U8(v)     => ("UINT8", serde_json::json!(v)),
            InferData::U16(v)    => ("UINT16", serde_json::json!(v)),
            InferData::U64(v)    => ("UINT64", serde_json::json!(v)),
            InferData::I8(v)     => ("INT8", serde_json::json!(v)),
            InferData::I16(v)    => ("INT16", serde_json::json!(v)),
            InferData::I32(v)    => ("INT32", serde_json::json!(v)),
            InferData::I64(v)    => ("INT64", serde_json::json!(v)),
            InferData::F32(v)    => ("FP32", serde_json::json!(v)),
            InferData::F64(v)    => ("FP64", serde_json::json!(v)),
            InferData::String(v) => ("STRING", serde_json::json!(v)),
            InferData::Bf16(v)   => ("BF16", serde_json::json!(v)),
        };

        InferInputPayload {
            name: &infer_input.input_name,
            shape: infer_input.input_shape.clone(),
            datatype,
            data: data_json,
        }
    }

    pub async fn infer(&self, inputs: Vec<InferInput>, model_name: &str) -> Result<InferResponse, TrustonError> {
        let url = format!("{}/v2/models/{}/infer", self.base_url, model_name);

        let input_payloads: Vec<_> = inputs
            .iter()
            .map(|inp| self.convert_input(inp))
            .collect();

        let request = InferRequest {
            inputs: input_payloads
        };

        let resp = self.http
        .post(&url)
        .json(&request)
        .send()
        .await?;

        let status = resp.status();

        if !status.is_success() {
            let error_body = resp.text().await.unwrap_or_else(|_| "Unknown error body".to_string());
            return Err(TrustonError::InferRequestError(error_body));
        }
    
        let response_struct: InferResponse = resp
            .json::<InferResponse>()
            .await
            .map_err(|e| TrustonError::InferParseError(e.to_string()))?; 

        Ok(response_struct)
    }
}


