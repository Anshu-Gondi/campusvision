use axum::{ extract::{ Multipart, Path, Query }, http::StatusCode, response::IntoResponse, Json };
use serde::{ Deserialize };

use crate::cctv::api::{
    process_frame_rust,
    get_tracked_faces_rust,
    clear_daily_rust,
    clear_camera,
    CctvResult,
};

/// Query parameters for processing a frame
#[derive(Deserialize)]
pub struct ProcessQuery {
    pub role: String,
    pub camera_id: String,
    pub min_confidence: Option<f32>,
    pub min_track_hits: Option<u32>,
}

/// Handler to process a CCTV frame
pub async fn process_frame_route(
    Query(q): Query<ProcessQuery>,
    mut multipart: Multipart
) -> impl IntoResponse {
    let mut frame_bytes: Option<Vec<u8>> = None;
    let mut model_path: Option<String> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        match field.name() {
            Some("frame") => {
                frame_bytes = field
                    .bytes().await
                    .ok()
                    .map(|b| b.to_vec());
            }
            Some("model_path") => {
                model_path = field.text().await.ok();
            }
            _ => {}
        }
    }

    let frame = match frame_bytes {
        Some(f) => f,
        None => {
            return (StatusCode::BAD_REQUEST, "frame is required").into_response();
        }
    };

    let frame_clone = frame.clone();
    let role = q.role.clone();
    let camera_id = q.camera_id.clone();
    let min_conf = q.min_confidence.unwrap_or(0.75);
    let min_hits = q.min_track_hits.unwrap_or(5);
    let model_path = model_path.clone();

    let result = tokio::task::spawn_blocking(move || {
        process_frame_rust(
            &frame_clone,
            &role,
            &camera_id,
            min_conf,
            min_hits,
        )
    }).await;

    match result {
        Ok(Ok(res)) => Json(res).into_response(),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Join error: {}", e)).into_response(),
    }
}

/// GET /cctv/tracks/:role/:camera_id
pub async fn get_tracks_route(Path((role, camera_id)): Path<
    (String, String)
>) -> Json<Vec<CctvResult>> {
    Json(get_tracked_faces_rust(&role, &camera_id))
}

/// POST /cctv/clear_daily
pub async fn clear_daily_route() -> StatusCode {
    clear_daily_rust();
    StatusCode::OK
}

/// POST /cctv/clear_camera/:role/:camera_id
pub async fn clear_camera_route(Path((role, camera_id)): Path<(String, String)>) -> StatusCode {
    clear_camera(&role, &camera_id);
    StatusCode::OK
}
