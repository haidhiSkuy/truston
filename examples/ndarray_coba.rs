use ndarray::ArrayD;
use ndarray::IxDyn;
use truston::client::io::InferInput;

fn main() {
    // bikin tensor shape (2,10) → 2x10 matrix
    let arr: ArrayD<f32> = ArrayD::zeros(IxDyn(&[2, 10]));

    let infer_input = InferInput::from_ndarray("dynamic_input", arr);

    println!("{:#?}", infer_input.input_data);
}
