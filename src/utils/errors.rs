//! Error types for Truston operations.
//!
//! This module defines all error types that can occur when using the Truston client.

use std::fmt;

/// The main error type for Truston operations.
///
/// All operations in Truston return `Result<T, TrustonError>` for comprehensive
/// error handling.
///
/// # Examples
///
/// ```no_run
/// use truston::client::triton_client::TritonRestClient;
/// use truston::utils::errors::TrustonError;
///
/// #[tokio::main]
/// async fn main() {
///     let client = TritonRestClient::new("http://localhost:8000");
///     
///     match client.is_server_live().await {
///         Ok(_) => println!("Success!"),
///         Err(TrustonError::Http(msg)) => {
///             eprintln!("Network error: {}", msg);
///         }
///         Err(TrustonError::HttpErrorResponse(code, msg)) => {
///             eprintln!("Server returned {}: {}", code, msg);
///         }
///         Err(e) => {
///             eprintln!("Other error: {:?}", e);
///         }
///     }
/// }
/// ```
/// 
/// 
#[derive(Debug)]
pub enum TrustonError {
    /// HTTP connection or network error.
    ///
    /// This error occurs when the request cannot be sent to the server,
    /// typically due to network issues, DNS failures, or connection timeouts.
    ///
    /// # Example
    ///
    /// ```
    /// use truston::utils::errors::TrustonError;
    ///
    /// let error = TrustonError::Http("Connection refused".to_string());
    /// println!("{:?}", error);
    /// ```
    Http(reqwest::Error),

    // HTTP error response from the server.
    ///
    /// This error occurs when the server returns a non-success status code
    /// (4xx or 5xx). The tuple contains the status code and error message.
    ///
    /// # Fields
    ///
    /// * `0` - HTTP status code (e.g., 404, 500)
    /// * `1` - Error message from the server
    ///
    /// # Example
    ///
    /// ```
    /// use truston::utils::errors::TrustonError;
    ///
    /// let error = TrustonError::ServerError(
    ///     404,
    ///     "Model not found".to_string()
    /// );
    /// 
    /// if let TrustonError::ServerError(code, msg) = error {
    ///     println!("Server error {}: {}", code, msg);
    /// }
    /// ```
    ServerError { status: u16, message: String },
    
    // Inference request was rejected by the server.
    ///
    /// This error occurs when the server rejects the inference request,
    /// typically due to invalid inputs, model errors, or server configuration issues.
    ///
    /// # Example
    ///
    /// ```
    /// use truston::utils::errors::TrustonError;
    ///
    /// let error = TrustonError::InferenceError(
    ///     "Invalid input shape".to_string()
    /// );
    /// ```
    InferenceError(String),

    /// Failed to parse the inference response.
    ///
    /// This error occurs when the response from the server cannot be parsed
    /// into the expected format, typically indicating a protocol mismatch or
    /// malformed response.
    ///
    /// # Example
    ///
    /// ```
    /// use truston::utils::errors::TrustonError;
    ///
    /// let error = TrustonError::ParseError(
    ///     "Expected JSON array".to_string()
    /// );
    /// ```
    ParseError(String),
}

impl fmt::Display for TrustonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(e) => write!(f, "HTTP error: {}", e),
            Self::ServerError { status, message } => 
                write!(f, "Server error {}: {}", status, message),
            Self::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl From<reqwest::Error> for TrustonError {
    fn from(e: reqwest::Error) -> Self {
        Self::Http(e)
    }
}

impl std::error::Error for TrustonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Http(e) => Some(e),
            _ => None,
        }
    }
}