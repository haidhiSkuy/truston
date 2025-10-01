# Truston

[![Crates.io](https://img.shields.io/crates/v/truston.svg)](https://crates.io/crates/truston)
[![Documentation](https://docs.rs/truston/badge.svg)](https://docs.rs/truston)
[![License](https://img.shields.io/crates/l/truston.svg)](LICENSE)

A high-performance Rust client library for [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).

Truston provides a type-safe, ergonomic interface for communicating with Triton Inference Server via its REST API, supporting multiple data types, seamless NDArray integration, and async operations.

## Features

- 🚀 **Type-safe inference** - Strongly-typed input/output handling with compile-time guarantees
- 🎯 **Multiple data types** - Support for all Triton data types (INT8-64, UINT8-64, FP32/64, BOOL, STRING, BF16)
- 🔢 **NDArray integration** - Direct conversion between `ndarray::ArrayD` and Triton tensors
- ⚡ **Async/await** - Built on `tokio` for efficient concurrent operations
- 🛡️ **Error handling** - Comprehensive error types with context
- 📊 **Production-ready** - Includes logging, tracing, and comprehensive tests

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
truston = "0.1.0"
tokio = { version = "1.47", features = ["full"] }
ndarray = "0.16"
```

## Quick Start

```rust
use truston::client::triton_client::TritonRestClient;
use truston::client::io::InferInput;
use ndarray::ArrayD;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client
    let client = TritonRestClient::new("http://localhost:8000");
    
    // Check if server is alive
    client.is_server_live().await?;
    
    // Prepare input data
    let input_data: ArrayD<f32> = ArrayD::zeros(ndarray::IxDyn(&[1, 224, 224, 3]));
    let input = InferInput::from_ndarray("input", input_data);
    
    // Run inference
    let results = client.infer(vec![input], "resnet50").await?;
    
    // Access results
    for output in results.outputs {
        println!("Output: {} with shape {:?}", output.name, output.shape);
        
        if let Some(vec) = output.data.as_f32_vec() {
            println!("First 5 values: {:?}", &vec[..5]);
        }
    }
    
    Ok(())
}
```

## Usage Examples

### Creating Inputs from NDArray

```rust
use truston::client::io::InferInput;
use ndarray::array;

// From a 2D array
let arr = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
let input = InferInput::from_ndarray("my_input", arr);
```

### Creating Inputs from Raw Vectors

```rust
use truston::client::io::{InferInput, DataType};

// For float32 data
let data = DataType::F32(vec![1.0, 2.0, 3.0, 4.0]);
let input = InferInput::new(
    "my_input".to_string(),
    vec![2, 2], // shape
    data
);

// For int64 data
let data = DataType::I64(vec![1, 2, 3, 4, 5, 6]);
let input = InferInput::new(
    "input_ids".to_string(),
    vec![1, 6],
    data
);
```

### Multi-Input Models

```rust
use truston::client::triton_client::TritonRestClient;
use truston::client::io::InferInput;
use ndarray::ArrayD;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = TritonRestClient::new("http://localhost:8000");
    
    // BERT-style model with multiple inputs
    let input_ids: ArrayD<i64> = ArrayD::zeros(ndarray::IxDyn(&[1, 128]));
    let attention_mask: ArrayD<i64> = ArrayD::ones(ndarray::IxDyn(&[1, 128]));
    
    let inputs = vec![
        InferInput::from_ndarray("input_ids", input_ids),
        InferInput::from_ndarray("attention_mask", attention_mask),
    ];
    
    let results = client.infer(inputs, "bert_model").await?;
    
    Ok(())
}
```

### Handling Outputs

```rust
// Convert to vector
if let Some(vec) = output.data.as_f32_vec() {
    println!("F32 output: {:?}", vec);
}

// Convert to ndarray
if let Some(arr) = output.data.to_ndarray_f32(&output.shape) {
    println!("Array shape: {:?}", arr.shape());
    println!("Max value: {:?}", arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
}

// Handle different types
match &output.data {
    DataType::F32(v) => println!("Float32: {} values", v.len()),
    DataType::I64(v) => println!("Int64: {} values", v.len()),
    DataType::String(v) => println!("Strings: {:?}", v),
    _ => println!("Other type"),
}
```

### Error Handling

```rust
use truston::utils::errors::TrustonError;

match client.infer(inputs, "my_model").await {
    Ok(results) => {
        println!("Success! Got {} outputs", results.outputs.len());
    }
    Err(TrustonError::Http(msg)) => {
        eprintln!("Connection error: {}", msg);
    }
    Err(TrustonError::HttpErrorResponse(code, body)) => {
        eprintln!("Server error {}: {}", code, body);
    }
    Err(TrustonError::InferRequestError(msg)) => {
        eprintln!("Inference failed: {}", msg);
    }
    Err(TrustonError::InferParseError(msg)) => {
        eprintln!("Parse error: {}", msg);
    }
}
```

## Supported Data Types

| Rust Type | Triton Type | DataType Variant |
|-----------|-------------|------------------|
| `bool` | BOOL | `DataType::Bool` |
| `u8` | UINT8 | `DataType::U8` |
| `u16` | UINT16 | `DataType::U16` |
| `u64` | UINT64 | `DataType::U64` |
| `i8` | INT8 | `DataType::I8` |
| `i16` | INT16 | `DataType::I16` |
| `i32` | INT32 | `DataType::I32` |
| `i64` | INT64 | `DataType::I64` |
| `f32` | FP32 | `DataType::F32` |
| `f64` | FP64 | `DataType::F64` |
| `String` | STRING | `DataType::String` |
| `u16` (raw) | BF16 | `DataType::Bf16` |

## Requirements

- Rust 1.70 or later
- Triton Inference Server (any version supporting v2 REST API)

## Running Tests

Some tests require a running Triton server:

```bash
# Run all tests except integration tests
cargo test

# Run integration tests (requires Triton server at localhost:50000)
cargo test -- --ignored
```

## Examples

Check the `examples/` directory for more usage examples:

```bash
# Check server connection
cargo run --example client_connect

# Run inference test
cargo run --example infer_test

# NDArray examples
cargo run --example ndarray_coba
```

## Documentation

For detailed API documentation, run:

```bash
cargo doc --open
```

Or visit [docs.rs/truston](https://docs.rs/truston).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Licensed:
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)


## Acknowledgments

- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- Built with [reqwest](https://github.com/seanmonstar/reqwest), [tokio](https://tokio.rs/), and [ndarray](https://github.com/rust-ndarray/ndarray)

## Resources

- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
- [Triton REST API Reference](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md)