use axum::{
    extract::{ State, Json },
    routing::post,
    Router,
    http::StatusCode,
    response::IntoResponse,
};
use serde::{ Deserialize, Serialize };
use std::sync::Arc;

use crate::app::AppState;

use super::detect::{ detect_and_embed_rust, detect_and_add_person_rust };
use super::index::{ search_person_rust, can_reenroll_rust };
use super::recognition::{ verify_face_onnx, detect_emotion_onnx };

// ── Shared API Error ─────────────────────────────────────────────────────

pub type ApiResult<T> = Result<Json<T>, ApiErr>;

#[derive(Debug)]
pub struct ApiErr {
    pub status: StatusCode,
    pub message: String,
}

#[derive(Serialize)]
pub struct ApiError {
    pub error: String,
}

impl IntoResponse for ApiErr {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(ApiError {
                error: self.message,
            }),
        ).into_response()
    }
}

// ── DTOs ────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct DetectEmbedRequest {
    pub image: Vec<u8>,
    pub model_path: Option<String>,
}

#[derive(Serialize)]
pub struct DetectEmbedResponse {
    pub found: bool,
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct AddPersonRequest {
    pub image: Vec<u8>,
    pub name: String,
    pub person_id: u64,
    pub roll_no: String,
    pub role: String,
}

#[derive(Serialize)]
pub struct AddPersonResponse {
    pub id: usize,
}

#[derive(Deserialize)]
pub struct SearchPersonRequest {
    pub embedding: Vec<f32>,
    pub role: String,
    pub k: usize,
}

#[derive(Serialize)]
pub struct SearchPersonResponse {
    pub results: Vec<(usize, f32)>,
}

#[derive(Deserialize)]
pub struct CanReenrollRequest {
    pub embedding: Vec<f32>,
    pub person_id: u64,
    pub role: String,
}

#[derive(Serialize)]
pub struct CanReenrollResponse {
    pub can: bool,
}

#[derive(Deserialize)]
pub struct VerifyAttendanceRequest {
    pub user_id: u64,
    pub embedding: Vec<f32>,
    pub role: String,
}

#[derive(Serialize)]
pub struct VerifyAttendanceResponse {
    pub status: String,
}

#[derive(Deserialize)]
pub struct VerifyFaceRequest {
    pub image: Vec<u8>,
    pub known_embedding: Vec<f32>,
}

#[derive(Serialize)]
pub struct VerifyFaceResponse {
    pub similarity: f32,
}

#[derive(Deserialize)]
pub struct DetectEmotionRequest {
    pub image: Vec<u8>,
}

#[derive(Serialize)]
pub struct DetectEmotionResponse {
    pub emotion: i64,
}

// ── Handlers ────────────────────────────────────────────────────────────

pub async fn detect_and_embed_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DetectEmbedRequest>
) -> ApiResult<DetectEmbedResponse> {
    let mut redis = state.redis.lock().await;

    if !state.security.allow_request(&mut redis, "detect_embed").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let result = detect_and_embed_rust(&payload.image, payload.model_path.as_deref()).map_err(
        |e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        }
    )?;

    Ok(
        Json(DetectEmbedResponse {
            found: result.found,
            bbox: result.bbox,
            embedding: result.embedding,
        })
    )
}

pub async fn detect_and_add_person_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AddPersonRequest>
) -> ApiResult<AddPersonResponse> {
    let mut redis = state.redis.lock().await;

    if !state.security.allow_request(&mut redis, "add_person").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let id = detect_and_add_person_rust(
        &payload.image,
        payload.name,
        payload.person_id,
        payload.roll_no,
        payload.role
    ).map_err(|e| ApiErr {
        status: StatusCode::BAD_REQUEST,
        message: e.to_string(),
    })?;

    Ok(Json(AddPersonResponse { id }))
}

pub async fn search_person_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchPersonRequest>
) -> ApiResult<SearchPersonResponse> {
    let mut redis = state.redis.lock().await;

    if !state.security.allow_request(&mut redis, "search_person").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let results = search_person_rust(payload.embedding, payload.role, payload.k).map_err(
        |e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        }
    )?;

    Ok(Json(SearchPersonResponse { results }))
}

pub async fn can_reenroll_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CanReenrollRequest>
) -> ApiResult<CanReenrollResponse> {
    let mut redis = state.redis.lock().await;

    if !state.security.allow_request(&mut redis, "can_reenroll").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let can = can_reenroll_rust(payload.embedding, payload.person_id, payload.role).map_err(
        |e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        }
    )?;

    Ok(Json(CanReenrollResponse { can }))
}

pub async fn verify_attendance_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<VerifyAttendanceRequest>
) -> ApiResult<VerifyAttendanceResponse> {
    let user_key = payload.user_id.to_string();
    let mut redis = state.redis.lock().await;

    // 1️⃣ Already marked
    if state.attendance.is_recent(&mut redis, &user_key).await {
        return Ok(
            Json(VerifyAttendanceResponse {
                status: "already_marked".into(),
            })
        );
    }

    // 2️⃣ Brute-force protection
    state.security.validate_attempt(&mut redis, &user_key).await.map_err(|e| ApiErr {
        status: StatusCode::FORBIDDEN,
        message: e,
    })?;

    // 3️⃣ Location verification (fail closed)
    let django = state.django.clone();
    let key = user_key.clone();

    let location_ok = tokio::task
        ::spawn_blocking(move || { django.is_location_valid(&key) }).await
        .unwrap_or(false);

    if !location_ok {
        return Err(ApiErr {
            status: StatusCode::FORBIDDEN,
            message: "User not at required location".into(),
        });
    }

    // 4️⃣ Mark attendance
    state.attendance.mark(&mut redis, &user_key).await;

    Ok(
        Json(VerifyAttendanceResponse {
            status: "marked".into(),
        })
    )
}

pub async fn verify_face_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<VerifyFaceRequest>
) -> ApiResult<VerifyFaceResponse> {
    let mut redis = state.redis.lock().await;

    if !state.security.allow_request(&mut redis, "verify_face").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let similarity = tokio::task
        ::spawn_blocking(move || {
            verify_face_onnx(payload.image, &payload.known_embedding)
        }).await
        .map_err(|_| ApiErr {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "Inference task panicked".into(),
        })?
        .map_err(|e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        })?;

    Ok(Json(VerifyFaceResponse { similarity }))
}

pub async fn detect_emotion_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DetectEmotionRequest>
) -> ApiResult<DetectEmotionResponse> {
    let mut redis = state.redis.lock().await;

    if !state.security.allow_request(&mut redis, "detect_emotion").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let emotion = tokio::task
        ::spawn_blocking(move || { detect_emotion_onnx(payload.image) }).await
        .map_err(|_| ApiErr {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "Inference task panicked".into(),
        })?
        .map_err(|e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        })?;

    Ok(Json(DetectEmotionResponse { emotion }))
}

// ── Routes ──────────────────────────────────────────────────────────────

pub fn face_routes(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/face/detect-embed", post(detect_and_embed_handler))
        .route("/face/add-person", post(detect_and_add_person_handler))
        .route("/face/search", post(search_person_handler))
        .route("/face/can-reenroll", post(can_reenroll_handler))
        .route("/face/verify", post(verify_face_handler))
        .route("/face/emotion", post(detect_emotion_handler))
        .route("/attendance/verify", post(verify_attendance_handler))
        .with_state(state)
}
