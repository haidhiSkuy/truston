use std::sync::Arc;
use ndarray::{ArrayD, IxDyn};
use tokio;
use truston::client::triton_client::TritonRestClient;
use truston::client::io::InferInput;
use truston::utils::errors::TrustonError;

async fn run_infer(client: Arc<TritonRestClient>) -> Result<(), TrustonError> {
    let arr_ids: ArrayD<i64> = ArrayD::zeros(IxDyn(&[32, 128]));
    let input_ids = InferInput::from_ndarray("input_ids", arr_ids);

    let arr_attention_mask: ArrayD<i64> = ArrayD::zeros(IxDyn(&[32, 128]));
    let input_attention_mask = InferInput::from_ndarray("attention_mask", arr_attention_mask);

    let inputs = vec![input_ids, input_attention_mask];

    let _ = client.infer(inputs, "hierarchical_clf").await?;
    Ok(())
}


#[tokio::main]
async fn main() {
    let my_client = Arc::new(TritonRestClient::new("http://localhost:50000"));

    let tasks: Vec<_> = (0..2)
        .map(|_| {
            let client = my_client.clone();
            tokio::spawn(async move {
                if let Err(e) = run_infer(client).await {
                    eprintln!("Error: {:?}", e);
                }
            })
        })
        .collect();

    futures::future::join_all(tasks).await;
}