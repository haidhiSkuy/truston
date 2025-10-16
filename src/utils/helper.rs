pub fn urljoin(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    format!("{}/{}", base, path)
}

#[cfg(test)]
mod tests {
    use super::*; 
    #[test]
    fn test_urljoin1() {
        let base = "http://localhost:3000";
        let path = "/this/is/endpoint";
        let url = urljoin(base, path);
        assert_eq!(url, "http://localhost:3000/this/is/endpoint");
    }
    
    #[test]
    fn test_urljoin2() {
        let base = "http://localhost:3000/";
        let path = "/this/is/endpoint";
        let url = urljoin(base, path);
        assert_eq!(url, "http://localhost:3000/this/is/endpoint");
    }
}
