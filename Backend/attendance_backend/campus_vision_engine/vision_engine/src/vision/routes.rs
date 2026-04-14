/// vision/routes.rs  (replaces existing)
///
/// Changes vs original:
///   1. Direct recognition handlers acquire a `scheduler.acquire_direct()`
///      permit before doing any work. This guarantees direct calls always
///      get a slot ahead of CCTV background frames.
///   2. LAST_EMBEDDING global Mutex replaced with per-user key in
///      AppState.last_embeddings (DashMap<String, Vec<f32>>).
///      Key = user_id or "detect:{ip}" for anonymous detect calls.
///   3. verify_attendance_handler flow preserved exactly.

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::app::AppState;
use intelligence_core::embeddings::add_face_embedding;
use intelligence_core::utils::cosine_similarity;

use super::detect::detect_and_embed_rust;
use super::index::search_person_rust;
use super::recognition::{verify_face_with_pool, detect_emotion_with_pool};

// ── Shared error type ─────────────────────────────────────────────────────────

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

impl axum::response::IntoResponse for ApiErr {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(ApiError { error: self.message })).into_response()
    }
}

// ── DTOs (unchanged from original) ───────────────────────────────────────────

#[derive(Deserialize)]
pub struct DetectEmbedRequest {
    pub image: Vec<u8>,
    pub user_id: Option<String>, // used for per-user dedup
    pub enrollment: Option<bool>,
}

#[derive(Serialize)]
pub struct DetectEmbedResponse {
    pub found: bool,
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct AddPersonRequest {
    pub school_id: String,
    pub embedding: Vec<f32>,
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
    pub school_id: String,
    pub embedding: Vec<f32>,
    pub role: String,
    pub k: usize,
}

#[derive(Serialize)]
pub struct SearchPersonResponse {
    pub results: Vec<(usize, f32)>,
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
    pub user_id: Option<String>,
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

#[derive(Deserialize)]
pub struct LoadDbRequest { pub school_id: String }
#[derive(Deserialize)]
pub struct InitDbRequest { pub school_id: String }
#[derive(Deserialize)]
pub struct SaveDbRequest { pub school_id: String }

#[derive(Serialize)]
struct ApiResponse { success: bool, message: String }

// ── Handlers ──────────────────────────────────────────────────────────────────

/// POST /face/detect-embed
///
/// Direct path: acquires DIRECT scheduler permit first.
/// Per-user replay guard via DashMap instead of global Mutex.
pub async fn detect_and_embed_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DetectEmbedRequest>,
) -> ApiResult<DetectEmbedResponse> {
    let mut redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "detect_embed").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    // ── Acquire direct permit (blocks until a slot is free, always ahead of CCTV) ──
    let _permit = state.scheduler.acquire_direct().await;

    let enrollment = payload.enrollment.unwrap_or(false);

    let result = detect_and_embed_rust(state.clone(), payload.image, None, enrollment)
        .await
        .map_err(|e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        })?;

    // Per-user replay guard (replaces global LAST_EMBEDDING mutex)
    if result.found {
        if let Some(embedding) = &result.embedding {
            let dedup_key = payload
                .user_id
                .clone()
                .unwrap_or_else(|| "anonymous".to_string());

            if let Some(prev) = state.last_embeddings.get(&dedup_key) {
                if cosine_similarity(&prev, embedding) >= 0.995 {
                    return Ok(Json(DetectEmbedResponse {
                        found: false,
                        bbox: None,
                        embedding: None,
                    }));
                }
            }
            state.last_embeddings.insert(dedup_key, embedding.clone());
        }
    }

    // Permit drops here — slot freed.
    Ok(Json(DetectEmbedResponse {
        found: result.found,
        bbox: result.bbox,
        embedding: result.embedding,
    }))
}

/// POST /face/verify
///
/// User's primary direct call. Gets direct permit.
pub async fn verify_face_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<VerifyFaceRequest>,
) -> ApiResult<VerifyFaceResponse> {
    let mut redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "verify_face").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    // ── Direct priority permit ────────────────────────────────────────────
    let _permit = state.scheduler.acquire_direct().await;

    let similarity = verify_face_with_pool(
        state.clone(),
        payload.image,
        payload.known_embedding,
        None,
        None,
    )
    .await
    .map_err(|e| ApiErr {
        status: StatusCode::BAD_REQUEST,
        message: e.to_string(),
    })?;

    Ok(Json(VerifyFaceResponse { similarity }))
}

/// POST /face/add-person — enrollment, direct lane.
pub async fn detect_and_add_person_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AddPersonRequest>,
) -> ApiResult<AddPersonResponse> {
    let mut redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "add_person").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let _permit = state.scheduler.acquire_direct().await;

    let id = add_face_embedding(
        &payload.school_id,
        payload.embedding,
        payload.name,
        payload.person_id,
        payload.roll_no,
        payload.role,
    )
    .map_err(|e| ApiErr {
        status: StatusCode::BAD_REQUEST,
        message: e.to_string(),
    })?;

    Ok(Json(AddPersonResponse { id }))
}

/// POST /face/search — direct lane.
pub async fn search_person_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchPersonRequest>,
) -> ApiResult<SearchPersonResponse> {
    let mut redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "search_person").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let _permit = state.scheduler.acquire_direct().await;

    let results = search_person_rust(&payload.school_id, payload.embedding, payload.role, payload.k)
        .map_err(|e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        })?;

    Ok(Json(SearchPersonResponse { results }))
}

/// POST /attendance/verify — direct lane.
pub async fn verify_attendance_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<VerifyAttendanceRequest>,
) -> ApiResult<VerifyAttendanceResponse> {
    let user_key = payload.user_id.to_string();
    let mut redis = state.redis.clone();
    let person_id = payload.user_id.to_string();

    if state.attendance.is_recent(&mut redis, &user_key, &person_id).await {
        return Ok(Json(VerifyAttendanceResponse { status: "already_marked".into() }));
    }

    state.security.validate_attempt(&mut redis, &user_key).await.map_err(|e| ApiErr {
        status: StatusCode::FORBIDDEN,
        message: e,
    })?;

    let _permit = state.scheduler.acquire_direct().await;

    let location_ok = state.django.is_location_valid(&user_key).await;
    if !location_ok {
        return Err(ApiErr {
            status: StatusCode::FORBIDDEN,
            message: "User not at required location".into(),
        });
    }

    let _ = state.attendance.mark(&mut redis, &user_key, &person_id).await;

    Ok(Json(VerifyAttendanceResponse { status: "marked".into() }))
}

/// POST /face/emotion — no priority permit needed (emotion model is separate pool).
pub async fn detect_emotion_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DetectEmotionRequest>,
) -> ApiResult<DetectEmotionResponse> {
    let mut redis = state.redis.clone();
    if !state.security.allow_request(&mut redis, "detect_emotion").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }
    let emotion = detect_emotion_with_pool(state.clone(), payload.image, None, None)
        .await
        .map_err(|e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        })?;
    Ok(Json(DetectEmotionResponse { emotion }))
}

// ── DB management handlers (unchanged logic, state flows through) ─────────────

pub async fn save_face_db_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SaveDbRequest>,
) -> impl IntoResponse {
    let local_dir = std::env::var("FACE_DB_PATH").unwrap_or_else(|_| "face_database".to_string());
    match state.face_db_backup.save(&payload.school_id, &local_dir).await {
        Ok(_) => (StatusCode::OK, Json(ApiResponse { success: true, message: "Saved".into() })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse { success: false, message: e.to_string() })),
    }
}

pub async fn init_face_db_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InitDbRequest>,
) -> impl IntoResponse {
    let local_dir = std::env::var("FACE_DB_PATH").unwrap_or_else(|_| "face_database".to_string());
    if let Err(e) = std::fs::create_dir_all(&local_dir) {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse { success: false, message: e.to_string() }));
    }
    if let Err(e) = crate::face_db::init_database_rust(&local_dir) {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse { success: false, message: e.to_string() }));
    }
    if let Err(e) = state.face_db_backup.save(&payload.school_id, &local_dir).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse { success: false, message: e.to_string() }));
    }
    (StatusCode::OK, Json(ApiResponse { success: true, message: "Initialized".into() }))
}

pub async fn load_face_db_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LoadDbRequest>,
) -> impl IntoResponse {
    let local_dir = std::env::var("FACE_DB_PATH").unwrap_or_else(|_| "face_database".to_string());
    match state.face_db_backup.load(&payload.school_id, &local_dir).await {
        Ok(_) => (StatusCode::OK, Json(ApiResponse { success: true, message: "Loaded".into() })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse { success: false, message: e.to_string() })),
    }
}

pub async fn inference_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.face_pool.is_ready() {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "warming")
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn face_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/face/detect-embed",  post(detect_and_embed_handler))
        .route("/face/add-person",    post(detect_and_add_person_handler))
        .route("/face/search",        post(search_person_handler))
        .route("/face/verify",        post(verify_face_handler))
        .route("/face/emotion",       post(detect_emotion_handler))
        .route("/attendance/verify",  post(verify_attendance_handler))
        .route("/face/load",          post(load_face_db_handler))
        .route("/face/save",          post(save_face_db_handler))
        .route("/face/init",          post(init_face_db_handler))
        .route("/health/inference",   get(inference_health))
}