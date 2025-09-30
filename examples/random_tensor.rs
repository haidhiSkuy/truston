use std::vec;

use truston::models::input_model::{InferData, InferInput}; 

fn main() {
    let dummy_input = InferData::F32(vec![5.0, 5.0, 5.0]);
    let _input = InferInput::new(
        "input_0".into(), 
        vec![1, 3], 
        dummy_input, 
    );
}