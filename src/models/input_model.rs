use ndarray::ArrayD;

#[derive(Debug)]
pub enum InferData {
    Bool(Vec<bool>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    String(Vec<String>),
    Bf16(Vec<u16>),
}

impl InferData {
    pub fn get_type_str(&self) -> &'static str {
        match self {
            InferData::Bool(_) => "BOOL",
            InferData::U8(_) => "UINT8",
            InferData::U16(_) => "UINT16",
            InferData::U64(_) => "UINT64",
            InferData::I8(_) => "INT8",
            InferData::I16(_) => "INT16",
            InferData::I32(_) => "INT32",
            InferData::I64(_) => "INT64",
            InferData::F32(_) => "FP32",
            InferData::F64(_) => "FP64",
            InferData::String(_) => "STRING",
            InferData::Bf16(_) => "BF16",
        }
    }
}

#[derive(Debug)]
pub struct InferInput {
    pub input_name: String,
    pub input_shape: Vec<usize>, 
    pub input_data: InferData,
}

impl InferInput {
    pub fn new(
        input_name: String,
        input_shape: Vec<usize>,
        input_data: InferData,
    ) -> Self {
        InferInput {
                input_name,
                input_shape,
                input_data,
            }
    }

    pub fn from_ndarray(name: impl Into<String>, arr: ArrayD<f32>) -> Self {
        let shape = arr.shape().to_vec();
        let (data, _) = arr.into_raw_vec_and_offset();
        Self {
            input_name: name.into(),
            input_shape: shape,
            input_data: InferData::F32(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_type_str() {
        assert_eq!(InferData::Bool(vec![true, false]).get_type_str(), "BOOL");
        assert_eq!(InferData::U8(vec![1, 2, 3]).get_type_str(), "UINT8");
        assert_eq!(InferData::I32(vec![42]).get_type_str(), "INT32");
        assert_eq!(InferData::F64(vec![3.14]).get_type_str(), "FP64");
        assert_eq!(InferData::String(vec!["hello".into()]).get_type_str(), "STRING");
        assert_eq!(InferData::Bf16(vec![0u16, 1u16]).get_type_str(), "BF16");
    }

    #[test]
    fn test_infer_input_new() {
        let input = InferInput::new(
            "my_input".to_string(),
            vec![1, 3, 224, 224],
            InferData::F32(vec![0.1, 0.2, 0.3]),
        );

        assert_eq!(input.input_name, "my_input");
        assert_eq!(input.input_shape, vec![1, 3, 224, 224]);

        match input.input_data {
            InferData::F32(values) => assert_eq!(values, vec![0.1, 0.2, 0.3]),
            _ => panic!("Expected F32 variant"),
        }
    }
}
