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
mod decision;
mod quality;
mod adaptive;

use crate::cctv::routes::*;
use crate::app::AppState;
use crate::vision::routes::face_routes;

use std::env;
use std::path::PathBuf;

/// Prepend the build OUT_DIR (where DLLs are copied) to PATH
fn setup_opencv_path() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let path_var = env::var("PATH").unwrap_or_default();
    let new_path = format!("{};{}", out_dir.display(), path_var);
    unsafe {
        env::set_var("PATH", new_path);
    }

    println!("✅ OpenCV DLLs path added: {}", out_dir.display());
}

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
    setup_opencv_path();

    std::panic::set_hook(
        Box::new(|panic_info| {
            println!("🔥 PANIC OCCURRED: {:?}", panic_info);
        })
    );

    dotenvy::dotenv().ok();

    CryptoProvider::install_default(aws_lc_rs::default_provider()).expect(
        "Failed to install aws-lc-rs crypto provider"
    );

    tracing_subscriber::fmt::init();

    // ✅ Initialize Global State
    let state = Arc::new(AppState::new().await);

    // ✅ Root Router with State
    let app = Router::new()
        .merge(cctv_router())
        .merge(face_routes())
        .route(
            "/health",
            get(|| async { "ok" })
        )
        .with_state(state.clone());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));

    tracing::info!("🚀 Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    let server = axum::serve(listener, app.into_make_service());

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

    Ok(())
}
