use axum::{
    Router,
    routing::{get, post},
};
use std::net::SocketAddr;
use std::sync::Arc;

mod cctv;
mod models;
mod preprocessing;
mod vision;
mod app;
mod security;
mod attendance;
mod django;

use crate::cctv::routes::*;
use crate::app::AppState;
use crate::vision::routes::*;

// Router setup (unchanged — this part is correct)
pub fn cctv_router() -> Router {
    Router::new()
        .route("/cctv/process-frame", post(process_frame_route))
        .route("/cctv/tracks/:role/:camera_id", get(get_tracks_route))
        .route("/cctv/clear-daily", post(clear_daily_route))
        .route("/cctv/clear-camera/:role/:camera_id", post(clear_camera_route))
}

// ── App Entry ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    // ✅ Shared global state
    let state = Arc::new(AppState::new().await);

    // ✅ Root router
    let app = Router::new()
        .nest("/", cctv_router())
        .nest("/", face_routes(state.clone())) // 🔥 NEW PART
        .route("/health", get(|| async { "ok" }));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("🚀 Server running at http://{}", addr);

    axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app.into_make_service(),
    )
    .await?;

    Ok(())
}