pub mod client;
pub mod utils;

use tracing_subscriber;

pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false) 
        .init();
}
