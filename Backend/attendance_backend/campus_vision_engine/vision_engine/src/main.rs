use axum::{ Router, routing::{ get, post } };
use std::net::SocketAddr;
use std::sync::Arc;
use rustls::crypto::{ CryptoProvider, aws_lc_rs };

mod cctv;
mod models;
mod preprocessing;
mod vision;
mod app;
mod service;
mod storage;
mod face_db;
mod scheduler;

use crate::cctv::routes::*;
use crate::app::AppState;
use crate::vision::routes::*;
use crate::scheduler::routes::*;

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
    dotenvy::dotenv().ok();

    CryptoProvider::install_default(aws_lc_rs::default_provider()).expect(
        "Failed to install aws-lc-rs crypto provider"
    );

    tracing_subscriber::fmt::init();

    // 🔥 Warm-up ONNX models before server starts
    if std::env::var("WARM_UP_MODELS").is_ok() {
        let model_paths: &[&'static str] = &[
            "models/facenet.onnx",
            "models/emotion.onnx",
        ];
        let _ = tokio::task::spawn_blocking(move || {
            crate::models::onnx_models::warm_up_onnx_models(model_paths)
        }).await?;
    }

    // ✅ Shared global state
    let state = Arc::new(AppState::new().await);

    // ✅ Root router
    let app = Router::new()
        .nest("/", cctv_router())
        .nest("/", face_routes(state.clone())) // 🔥 NEW PART
        .nest("/", scheduler_routes(state.clone())) // 🔥 NEW PART
        .route(
            "/health",
            get(|| async { "ok" })
        );

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("🚀 Server running at http://{}", addr);

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app.into_make_service()).await?;

    Ok(())
}
