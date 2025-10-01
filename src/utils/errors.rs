use std::fmt;

/// Custom error type for Truston client operations.
///
/// This enum represents all possible error conditions that can occur
/// while interacting with a Triton Inference Server via HTTP.
#[derive(Debug)]
pub enum TrustonError {
    /// Wraps any underlying HTTP/network error from [`reqwest`].
    ///
    /// This usually indicates a failed connection, timeout, or DNS issue.
    Http(reqwest::Error),

    /// Server responded with a non-success HTTP status code.
    ///
    /// This is used when the server returns a valid response
    /// but indicates an error condition (e.g., 400, 404, 500).
    ///
    /// - `status`: HTTP status code returned by the server.
    /// - `message`: Optional error message extracted from the response body.
    ServerError {
        /// HTTP status code returned by the server.
        status: u16,
        /// Error message returned by the server.
        message: String,
    },

    /// Error occurred during inference logic.
    ///
    /// This is not a transport or protocol error, but a logical
    /// issue during the inference process (e.g., invalid model input).
    InferenceError(String),

    /// Failed to parse server response or data.
    ///
    /// Typically occurs when the server returns malformed JSON
    /// or unexpected response fields.
    ParseError(String),
}

impl fmt::Display for TrustonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrustonError::Http(e) => write!(f, "HTTP error: {}", e),
            TrustonError::ServerError { status, message } => {
                write!(f, "Server error {}: {}", status, message)
            }
            TrustonError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            TrustonError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl From<reqwest::Error> for TrustonError {
    /// Converts a [`reqwest::Error`] into a [`TrustonError::Http`].
    fn from(e: reqwest::Error) -> Self {
        TrustonError::Http(e)
    }
}

impl std::error::Error for TrustonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TrustonError::Http(e) => Some(e),
            _ => None,
        }
    }
}
