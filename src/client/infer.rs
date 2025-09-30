use crate::client::triton_client::TritonRestClient;
use crate::models::input_model::{InferInput, InferData}; 

impl TritonRestClient {
    pub async fn infer(&self, infer_input: InferInput, model_name: &str) -> Result<String, reqwest::Error> {
        let url = format!("{}/v2/models/{}/infer", self.base_url, model_name);

        let (datatype, data_json) = match infer_input.input_data {
            InferData::Bool(v) => ("BOOL", serde_json::json!(v)),
            InferData::U8(v) => ("UINT8", serde_json::json!(v)),
            InferData::U16(v) => ("UINT16", serde_json::json!(v)),
            InferData::U64(v) => ("UINT64", serde_json::json!(v)),
            InferData::I8(v) => ("INT8", serde_json::json!(v)),
            InferData::I16(v) => ("INT16", serde_json::json!(v)),
            InferData::I32(v) => ("INT32", serde_json::json!(v)),
            InferData::I64(v) => ("INT64", serde_json::json!(v)),
            InferData::F32(v) => ("FP32", serde_json::json!(v)),
            InferData::F64(v) => ("FP64", serde_json::json!(v)),
            InferData::String(v) => ("STRING", serde_json::json!(v)),
            InferData::Bf16(v) => ("BF16", serde_json::json!(v)),
        };
        

        let payload = serde_json::json!({
            "inputs": [
                {
                    "name": infer_input.input_name,
                    "shape": infer_input.input_shape,
                    "datatype": datatype,
                    "data": data_json
                }
            ],
            "outputs": [
                { "name": "mobilenetv20_output_flatten0_reshape0" }
            ]
        });

        let resp = self.http
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        let body = resp.text().await?;
        Ok(body)
    }
}
