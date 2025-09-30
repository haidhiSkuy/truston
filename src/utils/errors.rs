#[derive(Debug)]
pub enum TrustonError {
    Http(String),
    HttpErrorResponse(u16, String), 
    InvalidResponse(String),
}