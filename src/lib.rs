//! # Truston
//!
//! A high-performance Rust client library for [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server).
//!
//! Truston provides a type-safe, ergonomic interface for communicating with Triton Inference Server
//! via its REST API. It supports multiple data types, seamless NDArray integration, and async operations.
//!
//! ## Features
//!
//! - **Type-safe inference**: Strongly-typed input/output handling with compile-time guarantees
//! - **Multiple data types**: Support for all Triton data types (INT8, INT16, INT32, INT64, UINT8, UINT16, UINT64, FP32, FP64, BOOL, STRING, BF16)
//! - **NDArray integration**: Direct conversion between `ndarray::ArrayD` and Triton tensors
//! - **Async/await**: Built on `tokio` for efficient concurrent operations
//! - **Error handling**: Comprehensive error types with context
//!
//! ## Quick Start
//!
//! ```no_run
//! use truston::client::triton_client::TritonRestClient;
//! use truston::client::io::InferInput;
//! use ndarray::ArrayD;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a client
//!     let client = TritonRestClient::new("http://localhost:8000");
//!     
//!     // Check if server is alive
//!     let is_alive = client.is_server_live().await?;
//!     println!("Server is alive: {}", is_alive);
//!     
//!     // Prepare input data
//!     let input_data: ArrayD<f32> = ArrayD::zeros(ndarray::IxDyn(&[1, 224, 224, 3]));
//!     let input = InferInput::from_ndarray("input", input_data);
//!     
//!     // Run inference
//!     let results = client.infer(vec![input], "my_model").await?;
//!     
//!     // Access results
//!     for output in results.outputs {
//!         println!("Output: {} with shape {:?}", output.name, output.shape);
//!     }
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Creating Inputs
//!
//! ### From NDArray
//!
//! ```
//! use truston::client::io::InferInput;
//! use ndarray::array;
//!
//! let arr = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
//! let input = InferInput::from_ndarray("my_input", arr);
//! ```
//!
//! ### From Raw Vectors
//!
//! ```
//! use truston::client::io::{InferInput, DataType};
//!
//! let data = DataType::F32(vec![1.0, 2.0, 3.0, 4.0]);
//! let input = InferInput::new(
//!     "my_input".to_string(),
//!     vec![2, 2], // shape
//!     data
//! );
//! ```
//!
//! ## Handling Outputs
//!
//! ```no_run
//! # use truston::client::triton_client::TritonRestClient;
//! # use truston::client::io::InferInput;
//! # use ndarray::ArrayD;
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let client = TritonRestClient::new("http://localhost:8000");
//! # let input_data: ArrayD<f32> = ArrayD::zeros(ndarray::IxDyn(&[1, 3]));
//! # let input = InferInput::from_ndarray("input", input_data);
//! let results = client.infer(vec![input], "my_model").await?;
//!
//! for output in results.outputs {
//!     // Convert to vector
//!     if let Some(vec) = output.data.as_f32_vec() {
//!         println!("F32 output: {:?}", vec);
//!     }
//!     
//!     // Convert to ndarray
//!     if let Some(arr) = output.data.to_ndarray_f32(&output.shape) {
//!         println!("Array shape: {:?}", arr.shape());
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Supported Data Types
//!
//! | Rust Type | Triton Type | DataType Variant |
//! |-----------|-------------|------------------|
//! | `bool` | BOOL | `DataType::Bool` |
//! | `u8` | UINT8 | `DataType::U8` |
//! | `u16` | UINT16 | `DataType::U16` |
//! | `u64` | UINT64 | `DataType::U64` |
//! | `i8` | INT8 | `DataType::I8` |
//! | `i16` | INT16 | `DataType::I16` |
//! | `i32` | INT32 | `DataType::I32` |
//! | `i64` | INT64 | `DataType::I64` |
//! | `f32` | FP32 | `DataType::F32` |
//! | `f64` | FP64 | `DataType::F64` |
//! | `String` | STRING | `DataType::String` |
//! | `u16` (raw) | BF16 | `DataType::Bf16` |
//!
//! ## Error Handling
//!
//! All operations return `Result<T, TrustonError>` for proper error handling:
//!
//! ```no_run
//! # use truston::client::triton_client::TritonRestClient;
//! # use truston::utils::errors::TrustonError;
//! # #[tokio::main]
//! # async fn main() {
//! let client = TritonRestClient::new("http://localhost:8000");
//!
//! match client.is_server_live().await {
//!     Ok(true) => println!("Server is ready"),
//!     Ok(false) => println!("Server is not ready"),
//!     Err(TrustonError::Http(msg)) => eprintln!("Connection error: {}", msg),
//!     Err(TrustonError::HttpErrorResponse(code, msg)) => {
//!         eprintln!("Server error {}: {}", code, msg)
//!     }
//!     Err(e) => eprintln!("Error: {:?}", e),
//! }
//! # }
//! ```
//!
//! ## Requirements
//!
//! - Rust 1.70 or later
//! - Triton Inference Server (any version supporting v2 REST API)
//!
//! ## License
//!
//! Licensed under either of Apache License, Version 2.0 or MIT license at your option.

pub mod client;
pub mod utils;

// Re-export commonly used items for convenience
pub use client::triton_client::{TritonClient, TritonRestClient};
pub use client::io::{DataType, InferInput, InferOutput, InferResults};
pub use utils::errors::TrustonError;

/// Initialize tracing subscriber for logging.
///
/// This sets up a formatted tracing subscriber with INFO level logging.
/// Call this once at the start of your application to enable logging.
///
/// # Example
///
/// ```
/// truston::init_tracing();
/// // Now tracing macros will output logs
/// ```
pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();
}