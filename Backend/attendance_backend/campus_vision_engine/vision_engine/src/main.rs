/// main.rs  (replaces existing)
///
/// Changes:
///   1. CCTV route paths updated to include school_id.
///   2. Single AppState constructed once, passed to all routers.
///   3. setup_opencv_path kept as-is.

use axum::{Router, routing::{get, post}};
use std::net::SocketAddr;
use std::sync::Arc;
use rustls::crypto::{CryptoProvider, aws_lc_rs};

mod scheduler;          // NEW
mod cctv;
mod models;
mod preprocessing;
mod vision;
mod app;
mod service;
mod storage;
mod face_db;
mod decision;
mod quality;
mod adaptive;

use crate::cctv::routes::*;
use crate::app::AppState;
use crate::vision::routes::face_routes;

use std::env;
use std::path::PathBuf;

fn setup_opencv_path() {
    if let Ok(out_dir) = env::var("OUT_DIR") {
        let out_dir = PathBuf::from(out_dir);
        let path_var = env::var("PATH").unwrap_or_default();
        let new_path = format!("{};{}", out_dir.display(), path_var);
        unsafe { env::set_var("PATH", new_path); }
        println!("✅ OpenCV DLLs path added: {}", out_dir.display());
    }
}

// ── CCTV Router ───────────────────────────────────────────────────────────────
//
// Routes updated:
//   - process-frame:  school_id is now a query param (easier for camera clients)
//   - tracks:         /:school_id/:role/:camera_id
//   - clear-camera:   /:school_id/:role/:camera_id

pub fn cctv_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/cctv/process-frame",                          post(process_frame_route))
        .route("/cctv/tracks/:school_id/:role/:camera_id",     get(get_tracks_route))
        .route("/cctv/clear-daily",                            post(clear_daily_route))
        .route("/cctv/clear-camera/:school_id/:role/:camera_id", post(clear_camera_route))
}

// ── App Entry ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_opencv_path();

    std::panic::set_hook(Box::new(|info| {
        println!("🔥 PANIC: {:?}", info);
    }));

    dotenvy::dotenv().ok();

    CryptoProvider::install_default(aws_lc_rs::default_provider())
        .expect("Failed to install crypto provider");

    tracing_subscriber::fmt::init();

    // ── Single AppState — shared by ALL routers ───────────────────────────
    //
    // Previously cctv/api.rs had its own GLOBAL_STATE that called
    // AppState::new() a second time (double Redis, double model pools).
    // Now there is exactly one AppState in the entire process.
    let state = Arc::new(AppState::new().await);

    let app = Router::new()
        .merge(cctv_router())
        .merge(face_routes())
        .route("/health", get(|| async { "ok" }))
        .with_state(state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000)); // 0.0.0.0 for cloud

    tracing::info!("🚀 Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let server = axum::serve(listener, app.into_make_service());

    let shutdown = async {
        tokio::signal::ctrl_c().await.expect("ctrl-c failed");
    };

    tokio::select! {
        _ = server => {},
        _ = shutdown => {
            println!("Shutting down gracefully...");
            state.face_pool.shutdown();
            state.emotion_pool.shutdown();
        }
    }

    Ok(())
}