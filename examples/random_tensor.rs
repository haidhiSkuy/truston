use std::vec;

use truston::models::{InferData, InferInput, DataType}; 

fn main() {
    let dummy_input = InferData::dummy_data(
        vec![1, 3, 224, 224], 
        DataType::FP32, 
    );

    let _input = InferInput {
        input_name: "input_0".into(), 
        input_shape: vec![1, 3, 224, 224], 
        input_type: DataType::FP64, 
        input_data: dummy_input, 
    };
}