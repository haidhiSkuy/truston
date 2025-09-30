use std::vec;

use tokio; 
use truston::client::triton_client::TritonRestClient; 
use ndarray::ArrayD;
use ndarray::IxDyn;
use truston::models::input_model::InferInput;


#[tokio::main]
async fn main(){

    let my_client = TritonRestClient::new("http://localhost:50000"); 
    // let is_alive = my_client.is_server_live().await;
    // match is_alive {
    //     Ok(_) => println!("Server is live!"),
    //     Err(e) => println!("error: {:#?}", e), 
    // } 

    let arr_ids: ArrayD<i64> = ArrayD::zeros(IxDyn(&[1, 128]));
    let input_ids = InferInput::from_ndarray("input_ids", arr_ids);

    let arr_attention_mask: ArrayD<i64> = ArrayD::zeros(IxDyn(&[1, 128]));
    let input_attention_mask = InferInput::from_ndarray("attention_mask", arr_attention_mask);


    let inputs = vec![input_ids, input_attention_mask];
    
    let res = my_client.infer(inputs, "hierarchical_clf").await;
    match res { 
        Ok(r) => println!("result: {:#?}", r), 
        Err(e) => println!("error: {:#?}", e),
    }


}