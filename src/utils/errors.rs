#[derive(Debug)]
pub enum TrustonError {
    Http(String),
    HttpErrorResponse(u16, String),
    InferRequestError(String), 
}

impl From<reqwest::Error> for TrustonError {
    fn from(err: reqwest::Error) -> Self {
        TrustonError::Http(err.to_string())
    }
}