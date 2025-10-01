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
    // vec
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

    // Ndarray
    pub fn to_ndarray_bool(&self, shape: &[usize]) -> Option<ArrayD<bool>> {
        if let DataType::Bool(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_u8(&self, shape: &[usize]) -> Option<ArrayD<u8>> {
        if let DataType::U8(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_u16(&self, shape: &[usize]) -> Option<ArrayD<u16>> {
        if let DataType::U16(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_u64(&self, shape: &[usize]) -> Option<ArrayD<u64>> {
        if let DataType::U64(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_i8(&self, shape: &[usize]) -> Option<ArrayD<i8>> {
        if let DataType::I8(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_i16(&self, shape: &[usize]) -> Option<ArrayD<i16>> {
        if let DataType::I16(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_i32(&self, shape: &[usize]) -> Option<ArrayD<i32>> {
        if let DataType::I32(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_i64(&self, shape: &[usize]) -> Option<ArrayD<i64>> {
        if let DataType::I64(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_f32(&self, shape: &[usize]) -> Option<ArrayD<f32>> {
        if let DataType::F32(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_f64(&self, shape: &[usize]) -> Option<ArrayD<f64>> {
        if let DataType::F64(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_string(&self, shape: &[usize]) -> Option<ArrayD<String>> {
        if let DataType::String(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
        } else {
            None
        }
    }
    pub fn to_ndarray_bf16(&self, shape: &[usize]) -> Option<ArrayD<u16>> {
        if let DataType::Bf16(v) = self {
            ArrayD::from_shape_vec(shape, v.clone()).ok()
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
    use ndarray::array;

    // ============ DataType Tests ============
    
    #[test]
    fn test_get_type_str() {
        assert_eq!(DataType::Bool(vec![true, false]).get_type_str(), "BOOL");
        assert_eq!(DataType::U8(vec![1, 2, 3]).get_type_str(), "UINT8");
        assert_eq!(DataType::U16(vec![1, 2]).get_type_str(), "UINT16");
        assert_eq!(DataType::U64(vec![1]).get_type_str(), "UINT64");
        assert_eq!(DataType::I8(vec![-1, 2]).get_type_str(), "INT8");
        assert_eq!(DataType::I16(vec![-100]).get_type_str(), "INT16");
        assert_eq!(DataType::I32(vec![42]).get_type_str(), "INT32");
        assert_eq!(DataType::I64(vec![1000]).get_type_str(), "INT64");
        assert_eq!(DataType::F32(vec![1.5]).get_type_str(), "FP32");
        assert_eq!(DataType::F64(vec![3.14]).get_type_str(), "FP64");
        assert_eq!(DataType::String(vec!["hello".into()]).get_type_str(), "STRING");
        assert_eq!(DataType::Bf16(vec![0u16, 1u16]).get_type_str(), "BF16");
        assert_eq!(DataType::Raw(serde_json::json!({})).get_type_str(), "none");
    }

    // ============ Vec Conversion Tests ============
    
    #[test]
    fn test_as_u8_vec() {
        let data = DataType::U8(vec![1, 2, 3, 4]);
        assert_eq!(data.as_u8_vec(), Some(vec![1, 2, 3, 4]));
        
        let wrong_type = DataType::I32(vec![1, 2, 3]);
        assert_eq!(wrong_type.as_u8_vec(), None);
    }

    #[test]
    fn test_as_i32_vec() {
        let data = DataType::I32(vec![-10, 0, 10]);
        assert_eq!(data.as_i32_vec(), Some(vec![-10, 0, 10]));
        
        let wrong_type = DataType::F32(vec![1.0]);
        assert_eq!(wrong_type.as_i32_vec(), None);
    }

    #[test]
    fn test_as_f32_vec() {
        let data = DataType::F32(vec![1.5, 2.5, 3.5]);
        assert_eq!(data.as_f32_vec(), Some(vec![1.5, 2.5, 3.5]));
    }

    #[test]
    fn test_as_f64_vec() {
        let data = DataType::F64(vec![3.14159, 2.71828]);
        assert_eq!(data.as_f64_vec(), Some(vec![3.14159, 2.71828]));
    }

    #[test]
    fn test_as_bool_vec() {
        let data = DataType::Bool(vec![true, false, true]);
        assert_eq!(data.as_bool_vec(), Some(vec![true, false, true]));
    }

    #[test]
    fn test_as_str_vec() {
        let data = DataType::String(vec!["hello".into(), "world".into()]);
        assert_eq!(data.as_str_vec(), Some(vec!["hello".to_string(), "world".to_string()]));
    }

    #[test]
    fn test_as_bf16_vec() {
        let data = DataType::Bf16(vec![100, 200, 300]);
        assert_eq!(data.as_bf16_vec(), Some(vec![100, 200, 300]));
    }

    #[test]
    fn test_all_signed_int_vecs() {
        assert_eq!(DataType::I8(vec![-1, 0, 1]).as_i8_vec(), Some(vec![-1, 0, 1]));
        assert_eq!(DataType::I16(vec![-100, 100]).as_i16_vec(), Some(vec![-100, 100]));
        assert_eq!(DataType::I64(vec![i64::MAX]).as_i64_vec(), Some(vec![i64::MAX]));
    }

    #[test]
    fn test_all_unsigned_int_vecs() {
        assert_eq!(DataType::U16(vec![1, 2, 3]).as_u16_vec(), Some(vec![1, 2, 3]));
        assert_eq!(DataType::U64(vec![u64::MAX]).as_u64_vec(), Some(vec![u64::MAX]));
    }

    // ============ NDArray Conversion Tests ============
    
    #[test]
    fn test_to_ndarray_f32() {
        let data = DataType::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let arr = data.to_ndarray_f32(&[2, 3]).unwrap();
        
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn test_to_ndarray_i32() {
        let data = DataType::I32(vec![1, 2, 3, 4]);
        let arr = data.to_ndarray_i32(&[2, 2]).unwrap();
        
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr[[0, 1]], 2);
        assert_eq!(arr[[1, 1]], 4);
    }

    #[test]
    fn test_to_ndarray_bool() {
        let data = DataType::Bool(vec![true, false, true, false]);
        let arr = data.to_ndarray_bool(&[2, 2]).unwrap();
        
        assert_eq!(arr[[0, 0]], true);
        assert_eq!(arr[[0, 1]], false);
    }

    #[test]
    fn test_to_ndarray_string() {
        let data = DataType::String(vec!["a".into(), "b".into(), "c".into()]);
        let arr = data.to_ndarray_string(&[3]).unwrap();
        
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr[[1]], "b");
    }

    #[test]
    fn test_to_ndarray_wrong_shape() {
        let data = DataType::F32(vec![1.0, 2.0, 3.0]);
        // Shape [2, 2] requires 4 elements, but we only have 3
        let result = data.to_ndarray_f32(&[2, 2]);
        assert!(result.is_none());
    }

    #[test]
    fn test_to_ndarray_wrong_type() {
        let data = DataType::I32(vec![1, 2, 3, 4]);
        // Trying to convert I32 to F32 ndarray
        let result = data.to_ndarray_f32(&[2, 2]);
        assert!(result.is_none());
    }

    #[test]
    fn test_to_ndarray_multidimensional() {
        let data = DataType::U8(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let arr = data.to_ndarray_u8(&[2, 2, 2]).unwrap();
        
        assert_eq!(arr.shape(), &[2, 2, 2]);
        assert_eq!(arr[[0, 0, 0]], 1);
        assert_eq!(arr[[1, 1, 1]], 8);
    }

    // ============ IntoInferData Trait Tests ============
    
    #[test]
    fn test_into_infer_data_bool() {
        let vec = vec![true, false];
        let data = vec.into_infer_data();
        assert!(matches!(data, DataType::Bool(_)));
    }

    #[test]
    fn test_into_infer_data_numeric_types() {
        assert!(matches!(vec![1u8].into_infer_data(), DataType::U8(_)));
        assert!(matches!(vec![1u16].into_infer_data(), DataType::U16(_)));
        assert!(matches!(vec![1u64].into_infer_data(), DataType::U64(_)));
        assert!(matches!(vec![1i8].into_infer_data(), DataType::I8(_)));
        assert!(matches!(vec![1i16].into_infer_data(), DataType::I16(_)));
        assert!(matches!(vec![1i32].into_infer_data(), DataType::I32(_)));
        assert!(matches!(vec![1i64].into_infer_data(), DataType::I64(_)));
        assert!(matches!(vec![1.0f32].into_infer_data(), DataType::F32(_)));
        assert!(matches!(vec![1.0f64].into_infer_data(), DataType::F64(_)));
    }

    #[test]
    fn test_into_infer_data_string() {
        let vec = vec!["test".to_string()];
        let data = vec.into_infer_data();
        assert!(matches!(data, DataType::String(_)));
    }

    // ============ InferInput Tests ============
    
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
    
    #[test]
    fn test_infer_input_from_ndarray_i32() {
        let arr = array![1, 2, 3, 4, 5, 6].into_dyn();
        let input = InferInput::from_ndarray("int_input", arr);
        
        assert_eq!(input.input_name, "int_input");
        assert_eq!(input.input_shape, vec![6]);
        
        let vec = input.input_data.as_i32_vec().unwrap();
        assert_eq!(vec, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_infer_input_from_ndarray_3d() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let arr = ArrayD::from_shape_vec(vec![2, 3, 4], data).unwrap();
        let input = InferInput::from_ndarray("3d_tensor", arr);
        
        assert_eq!(input.input_shape, vec![2, 3, 4]);
        assert_eq!(input.input_data.as_f64_vec().unwrap().len(), 24);
    }

    #[test]
    fn test_infer_input_string_name_conversion() {
        let arr = array![1.0f32].into_dyn();
        let input = InferInput::from_ndarray("literal_str", arr);
        assert_eq!(input.input_name, "literal_str");
        
        let input2 = InferInput::from_ndarray(String::from("string_type"), array![1.0f32].into_dyn());
        assert_eq!(input2.input_name, "string_type");
    }

    // ============ Edge Cases ============
    
    #[test]
    fn test_empty_vectors() {
        let data = DataType::F32(vec![]);
        assert_eq!(data.as_f32_vec(), Some(vec![]));
        
        let arr_result = data.to_ndarray_f32(&[0]);
        assert!(arr_result.is_some());
    }

    #[test]
    fn test_single_element() {
        let data = DataType::I32(vec![42]);
        let arr = data.to_ndarray_i32(&[1]).unwrap();
        assert_eq!(arr[[0]], 42);
    }

    #[test]
    fn test_large_shape() {
        let size = 1000;
        let data = DataType::U8((0..size).map(|x| (x % 256) as u8).collect());
        let arr = data.to_ndarray_u8(&[10, 10, 10]).unwrap();
        assert_eq!(arr.shape(), &[10, 10, 10]);
    }

    // ============ Type Safety Tests ============
    
    #[test]
    fn test_type_mismatch_returns_none() {
        let data = DataType::F32(vec![1.0, 2.0]);
        
        assert!(data.as_i32_vec().is_none());
        assert!(data.as_bool_vec().is_none());
        assert!(data.as_str_vec().is_none());
        assert!(data.to_ndarray_i32(&[2]).is_none());
    }

    #[test]
    fn test_cloning_independence() {
        let original = vec![1, 2, 3];
        let data = DataType::I32(original.clone());
        let cloned = data.as_i32_vec().unwrap();
        
        // Ensure the cloned vector is independent
        assert_eq!(cloned, vec![1, 2, 3]);
        assert_eq!(original, vec![1, 2, 3]);
    }
}