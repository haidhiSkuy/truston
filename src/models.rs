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
}