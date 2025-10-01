use std::sync::Arc;
use ndarray::{ArrayD, IxDyn};
use truston::client::io::InferInput;
use truston::utils::errors::TrustonError;
use truston::client::triton_client::{TritonRestClient}; 


#[tokio::test]
async fn test_server_live() -> Result<(), TrustonError> {
    let client = TritonRestClient::new("http://localhost:50000");
    let is_alive = client.is_server_live().await?;
    assert!(is_alive, "Expected server to be alive at http://localhost:50000");
    Ok(())
}

#[tokio::test]
async fn test_server_dead() {
    let client = TritonRestClient::new("http://127.0.0.1:9999"); // port dummy
    let result = client.is_server_live().await;
    assert!(
        result.is_err(),
        "Expected error when server is not reachable"
    );
}

async fn run_infer(client: Arc<TritonRestClient>) -> Result<(), TrustonError> {
    // dummy input_ids [32, 128]
    let arr_ids: ArrayD<i64> = ArrayD::zeros(IxDyn(&[32, 128]));
    let input_ids = InferInput::from_ndarray("input_ids", arr_ids);

    // dummy attention_mask [32, 128]
    let arr_attention_mask: ArrayD<i64> = ArrayD::zeros(IxDyn(&[32, 128]));
    let input_attention_mask = InferInput::from_ndarray("attention_mask", arr_attention_mask);

    let inputs = vec![input_ids, input_attention_mask];

    let result = client.infer(inputs, "hierarchical_clf").await?;

    assert!(
        !result.outputs.is_empty(),
        "Expected at least one output from inference"
    );

    Ok(())
}

/// Integration test: jalanin 2 concurrent infer request
#[tokio::test]
async fn test_concurrent_inference() -> Result<(), TrustonError> {
    let my_client = Arc::new(TritonRestClient::new("http://localhost:50000"));
    let tasks: Vec<_> = (0..5)
        .map(|_| {
            let client = my_client.clone();
            tokio::spawn(async move { run_infer(client).await })
        })
        .collect();
    let results = futures::future::join_all(tasks).await;
    for r in results {
        let inner = r.expect("Task panicked");
        assert!(inner.is_ok(), "Inference failed: {:?}", inner.err());
    }

    Ok(())
}
