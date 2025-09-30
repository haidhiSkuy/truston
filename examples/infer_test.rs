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

    let arr: ArrayD<f32> = ArrayD::zeros(IxDyn(&[1, 3, 224, 224]));
    let infer_input = InferInput::from_ndarray("data", arr);
    
    let res = my_client.infer(infer_input, "mobilenet").await;
    // match res { 
    //     Ok(r) => println!("result: {:#?}", r), 
    //     Err(e) => println!("error: {:#?}", e),
    // }


}