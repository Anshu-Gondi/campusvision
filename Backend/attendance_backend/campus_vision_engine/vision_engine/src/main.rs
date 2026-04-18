use axum::{Router, routing::{get, post}};
use std::net::SocketAddr;
use std::sync::Arc;
use rustls::crypto::{CryptoProvider, aws_lc_rs};

mod scheduler;
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

pub fn cctv_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/cctv/process-frame",                          post(process_frame_route))
        .route("/cctv/tracks/:school_id/:role/:camera_id",     get(get_tracks_route))
        .route("/cctv/clear-daily",                            post(clear_daily_route))
        .route("/cctv/clear-camera/:school_id/:role/:camera_id", post(clear_camera_route))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_opencv_path();

    std::panic::set_hook(Box::new(|info| {
        println!("🔥 PANIC: {:?}", info);
    }));

    CryptoProvider::install_default(aws_lc_rs::default_provider())
        .expect("Failed to install crypto provider");

    tracing_subscriber::fmt::init();

    // Build AppState once
    println!("🚀 Initializing AppState (models + connections)...");
    let state = Arc::new(AppState::new().await);
    println!("✅ AppState initialized successfully");

    let app = Router::new()
        .merge(cctv_router())
        .merge(face_routes())
        .route("/health", get(|| async { "ok" }))
        .with_state(state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));

    tracing::info!("🚀 Server running at http://{}", addr);
    println!("🚀 Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    let shutdown_signal = async {
        tokio::signal::ctrl_c().await.expect("failed to listen for ctrl-c");
        println!("🛑 Shutdown signal received...");
    };

    tokio::select! {
        _ = axum::serve(listener, app.into_make_service()) => {},
        _ = shutdown_signal => {
            println!("Shutting down gracefully...");
            
            // Clean shutdown of thread pools
            state.yunet_pool.shutdown();   // if it has this method
            state.face_pool.shutdown();
            state.emotion_pool.shutdown();
            
            // Give workers a moment to finish
            tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;
        }
    }

    Ok(())
}