use std::vec;

use truston::client::io::{DataType, InferInput};

fn main() {
    let dummy_input = DataType::F32(vec![5.0, 5.0, 5.0]);
    let _input = InferInput::new("input_0".into(), vec![1, 3], dummy_input);
}
