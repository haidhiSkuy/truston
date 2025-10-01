use std::fmt;

#[derive(Debug)]
pub enum TrustonError {
    Http(reqwest::Error),
    ServerError { status: u16, message: String },
    InferenceError(String),
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