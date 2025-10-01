use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

// ################ INPUT #######################
#[derive(Debug, Clone)]
pub enum DataType {
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
    Raw(serde_json::Value),
}

impl DataType {
    pub fn get_type_str(&self) -> &'static str {
        match self {
            DataType::Bool(_) => "BOOL",
            DataType::U8(_) => "UINT8",
            DataType::U16(_) => "UINT16",
            DataType::U64(_) => "UINT64",
            DataType::I8(_) => "INT8",
            DataType::I16(_) => "INT16",
            DataType::I32(_) => "INT32",
            DataType::I64(_) => "INT64",
            DataType::F32(_) => "FP32",
            DataType::F64(_) => "FP64",
            DataType::String(_) => "STRING",
            DataType::Bf16(_) => "BF16",
            DataType::Raw(_) => "none"
        }
    }
    pub fn as_u8_vec(&self) -> Option<Vec<u8>> {
        if let DataType::U8(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_u16_vec(&self) -> Option<Vec<u16>> {
        if let DataType::U16(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_u64_vec(&self) -> Option<Vec<u64>> {
        if let DataType::U64(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_i8_vec(&self) -> Option<Vec<i8>> {
        if let DataType::I8(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_i16_vec(&self) -> Option<Vec<i16>> {
        if let DataType::I16(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_i32_vec(&self) -> Option<Vec<i32>> {
        if let DataType::I32(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_i64_vec(&self) -> Option<Vec<i64>> {
        if let DataType::I64(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_f32_vec(&self) -> Option<Vec<f32>> {
        if let DataType::F32(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        if let DataType::F64(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_bool_vec(&self) -> Option<Vec<bool>> {
        if let DataType::Bool(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_bf16_vec(&self) -> Option<Vec<u16>> {
        if let DataType::Bf16(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
    pub fn as_str_vec(&self) -> Option<Vec<String>> {
        if let DataType::String(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }

}

pub trait IntoInferData {
    fn into_infer_data(self) -> DataType;
}

impl IntoInferData for Vec<bool> {
    fn into_infer_data(self) -> DataType {
        DataType::Bool(self)
    }
}
impl IntoInferData for Vec<u8> {
    fn into_infer_data(self) -> DataType {
        DataType::U8(self)
    }
}
impl IntoInferData for Vec<u16> {
    fn into_infer_data(self) -> DataType {
        DataType::U16(self)
    }
}
impl IntoInferData for Vec<u64> {
    fn into_infer_data(self) -> DataType {
        DataType::U64(self)
    }
}
impl IntoInferData for Vec<i8> {
    fn into_infer_data(self) -> DataType {
        DataType::I8(self)
    }
}
impl IntoInferData for Vec<i16> {
    fn into_infer_data(self) -> DataType {
        DataType::I16(self)
    }
}
impl IntoInferData for Vec<i32> {
    fn into_infer_data(self) -> DataType {
        DataType::I32(self)
    }
}
impl IntoInferData for Vec<i64> {
    fn into_infer_data(self) -> DataType {
        DataType::I64(self)
    }
}
impl IntoInferData for Vec<f32> {
    fn into_infer_data(self) -> DataType {
        DataType::F32(self)
    }
}
impl IntoInferData for Vec<f64> {
    fn into_infer_data(self) -> DataType {
        DataType::F64(self)
    }
}
impl IntoInferData for Vec<String> {
    fn into_infer_data(self) -> DataType {
        DataType::String(self)
    }
}


#[derive(Debug)]
pub struct InferInput {
    pub input_name: String,
    pub input_shape: Vec<usize>, 
    pub input_data: DataType,
}

impl InferInput {
    pub fn new(
        input_name: String,
        input_shape: Vec<usize>,
        input_data: DataType,
    ) -> Self {
        InferInput {
                input_name,
                input_shape,
                input_data,
            }
    }

    pub fn from_ndarray<T>(name: impl Into<String>, arr: ArrayD<T>) -> Self
    where
        T: Clone + 'static,
        Vec<T>: IntoInferData,
    {
        let shape = arr.shape().to_vec();
        let (data, _) = arr.into_raw_vec_and_offset();
        Self {
            input_name: name.into(),
            input_shape: shape,
            input_data: data.into_infer_data(),
        }
    }
}

// ######################## TRITON REQUEST #############################
#[derive(Serialize)]
pub struct InferRequest<'a, T> {
    pub inputs: Vec<InferInputPayload<'a, T>>,
}

#[derive(Serialize)]
pub struct InferInputPayload<'a, T> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub datatype: &'a str,
    pub data: T,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TritonServerResponse {
    pub name: String,
    pub shape: Vec<usize>,
    pub datatype: String,
    pub data: serde_json::Value,
}
#[derive(Debug, Deserialize, Clone)]
pub struct InferResponse {
    pub outputs: Vec<TritonServerResponse>,
}


// ####################### Output forwarded to user ######################
#[derive(Debug, Clone)]
pub struct InferOutput {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<usize>,
    pub data: DataType,
}

#[derive(Debug, Clone)]
pub struct InferResults {
    pub outputs: Vec<InferOutput>, 
}


// ######################## UNIT TEST ###################
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_type_str() {
        assert_eq!(DataType::Bool(vec![true, false]).get_type_str(), "BOOL");
        assert_eq!(DataType::U8(vec![1, 2, 3]).get_type_str(), "UINT8");
        assert_eq!(DataType::I32(vec![42]).get_type_str(), "INT32");
        assert_eq!(DataType::F64(vec![3.14]).get_type_str(), "FP64");
        assert_eq!(DataType::String(vec!["hello".into()]).get_type_str(), "STRING");
        assert_eq!(DataType::Bf16(vec![0u16, 1u16]).get_type_str(), "BF16");
    }

    #[test]
    fn test_infer_input_new() {
        let input = InferInput::new(
            "my_input".to_string(),
            vec![1, 3, 224, 224],
            DataType::F32(vec![0.1, 0.2, 0.3]),
        );

        assert_eq!(input.input_name, "my_input");
        assert_eq!(input.input_shape, vec![1, 3, 224, 224]);

        match input.input_data {
            DataType::F32(values) => assert_eq!(values, vec![0.1, 0.2, 0.3]),
            _ => panic!("Expected F32 variant"),
        }
    }
}
