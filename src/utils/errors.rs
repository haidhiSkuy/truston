#[derive(Debug)]
pub enum TrustonError {
    Http(String),
    InvalidResponse(String),
}
