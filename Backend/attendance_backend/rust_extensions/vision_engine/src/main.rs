use axum::{
    Router,
    routing::{get, post},
    Json,
    response::IntoResponse,
};
use std::net::SocketAddr;

mod cctv;
mod models;
mod preprocessing;

use crate::cctv::routes::*;

// Router setup (unchanged — this part is correct)
pub fn cctv_router() -> Router {
    Router::new()
        .route("/cctv/process-frame", post(process_frame_route))
        .route("/cctv/tracks/:role/:camera_id", get(get_tracks_route))
        .route("/cctv/clear-daily", post(clear_daily_route))
        .route("/cctv/clear-camera/:role/:camera_id", post(clear_camera_route))
        .route("/health", get(|| async { "ok" }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Optional: Add tracing for debugging (highly recommended)
    tracing_subscriber::fmt::init();

    let app = Router::new().nest("/", cctv_router());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("🚀 CCTV server running at http://{}", addr);

    // FIXED: Use axum::serve with TcpListener (Axum 0.7 standard)
    axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app.into_make_service(),
    )
    .await?;

    Ok(())
}