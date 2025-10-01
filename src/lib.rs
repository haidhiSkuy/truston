//! # Truston
//! 
//! A Rust client library for NVIDIA Triton Inference Server.
//!
//! ## Quick Start
//!
//! ```no_run
//! use truston::client::triton_client::TritonRestClient;
//!
//! #[tokio::main]
//! async fn main() {
//!     let client = TritonRestClient::new("http://localhost:8000");
//!     let is_alive = client.is_server_live().await.unwrap();
//!     println!("Server alive: {}", is_alive);
//! }
//! ```


pub mod client;
pub mod utils;

pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false) 
        .init();
}
