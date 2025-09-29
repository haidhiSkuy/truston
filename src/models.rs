use rand::Rng;
use rand::distributions::Alphanumeric; 

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    BOOL,
    UINT8,
    UINT16,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    FP32, 
    FP64,
    STRING,
    BF16,
}

impl DataType {
    fn type_str(&self) -> &'static str { 
        match self {
            DataType::BOOL => "BOOL",
            DataType::UINT8 => "UINT8",
            DataType::UINT16 => "UINT16",
            DataType::UINT64 => "UINT64", 
            DataType::INT8 => "INT8",
            DataType::INT16 => "INT16",
            DataType::INT32 => "INT32",
            DataType::INT64 => "INT64",
            DataType::FP32 => "FP32",
            DataType::FP64 => "FP64",
            DataType::STRING => "STRING",
            DataType::BF16 => "BF16",
        }
    }
}

#[derive(Debug)]
pub enum InferData {
    Bool(Vec<bool>),
    UINT8(Vec<u8>),
    UINT16(Vec<u16>),
    UINT64(Vec<u64>),
    INT8(Vec<i8>),
    INT16(Vec<i16>),
    INT32(Vec<i32>),
    INT64(Vec<i64>),
    FP32(Vec<f32>),
    FP64(Vec<f64>), 
    STRING(Vec<String>), 
    BF16(Vec<u16>),
}

impl InferData {
    pub fn dummy_data(shape: Vec<usize>, dtype: DataType) -> Self {
        let mut rng = rand::thread_rng();
        let numel: usize = shape.iter().product();
        
        let data = match dtype {
            DataType::BOOL => InferData::Bool((0..numel).map(|_| rng.gen_bool(0.5)).collect()),
            DataType::UINT8 => InferData::UINT8((0..numel).map(|_| rng.gen_range(0..5)).collect()),
            DataType::UINT16 => InferData::UINT16((0..numel).map(|_| rng.gen_range(0..5)).collect()),
            DataType::UINT64 => InferData::UINT64((0..numel).map(|_| rng.gen_range(0..5)).collect()),
            DataType::INT8 => InferData::INT8((0..numel).map(|_| rng.gen_range(0..5)).collect()),
            DataType::INT16 => InferData::INT16((0..numel).map(|_| rng.gen_range(0..5)).collect()),
            DataType::INT32 => InferData::INT32((0..numel).map(|_| rng.gen_range(0..5)).collect()),
            DataType::INT64 => InferData::INT64((0..numel).map(|_| rng.gen_range(0..5)).collect()),
            DataType::FP32 => InferData::FP32((0..numel).map(|_| rng.gen_range(0.0..5.0)).collect()),
            DataType::FP64 => InferData::FP64((0..numel).map(|_| rng.gen_range(0.0..5.0)).collect()),
            DataType::BF16 => InferData::BF16((0..numel).map(|_| rng.gen_range(0..5)).collect()),
            DataType::STRING => InferData::STRING(
                (0..numel)
                    .map(|_| {
                        (0..rng.gen_range(5..=10))
                            .map(|_| rng.sample(Alphanumeric) as char)
                            .collect::<String>()
                    })
                    .collect(),
            ),
        };
        data
    }
}

#[derive(Debug)]
pub struct InferInput {
    pub input_name: String,
    pub input_shape: Vec<usize>, 
    pub input_type: DataType, 
    pub input_data: InferData,
}