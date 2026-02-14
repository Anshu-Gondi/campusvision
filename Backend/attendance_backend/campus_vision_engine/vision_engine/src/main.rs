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
use crate::vision::routes::face_routes;
use crate::scheduler::routes::scheduler_routes;

// ── CCTV Router ─────────────────────────────────────────

pub fn cctv_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/cctv/process-frame", post(process_frame_route))
        .route("/cctv/tracks/:role/:camera_id", get(get_tracks_route))
        .route("/cctv/clear-daily", post(clear_daily_route))
        .route("/cctv/clear-camera/:role/:camera_id", post(clear_camera_route))
}

// ── App Entry ───────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 🔥 Panic Hook
    std::panic::set_hook(
        Box::new(|panic_info| {
            println!("🔥 PANIC OCCURRED: {:?}", panic_info);
        })
    );

    dotenvy::dotenv().ok();

    // 🔐 Install aws-lc-rs crypto
    CryptoProvider::install_default(aws_lc_rs::default_provider()).expect(
        "Failed to install aws-lc-rs crypto provider"
    );

    tracing_subscriber::fmt::init();

    // ✅ Initialize Global State
    let state = Arc::new(AppState::new().await);

    let shutdown_signal = async {
        tokio::signal::ctrl_c().await.expect("Failed to listen for shutdown");
    };

    tokio::select! {
        _ = server => {},
        _ = shutdown_signal => {
            println!("Shutting down gracefully...");
            state.face_pool.shutdown();
            state.emotion_pool.shutdown();
        }
    }

    // ✅ Root Router with State
    let app = Router::new()
        .merge(cctv_router())
        .merge(face_routes())
        .merge(scheduler_routes())
        .route(
            "/health",
            get(|| async { "ok" })
        )
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));

    tracing::info!("🚀 Server running at http://{}", addr);

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app.into_make_service()).await?;

    Ok(())
}
