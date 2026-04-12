use axum::{
    extract::{ State, Json },
    routing::{ post, get },
    Router,
    http::StatusCode,
    response::IntoResponse,
};
use serde::{ Deserialize, Serialize };
use std::sync::Arc;

use crate::app::AppState;
use intelligence_core::embeddings::add_face_embedding;

use super::detect::detect_and_embed_rust;
use super::index::{ search_person_rust };
use super::recognition::{ verify_face_with_pool, detect_emotion_with_pool };

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
pub struct LoadDbRequest {
    pub school_id: String,
}

#[derive(Deserialize)]
pub struct InitDbRequest {
    pub school_id: String,
}

#[derive(Deserialize)]
pub struct SaveDbRequest {
    pub school_id: String,
}

#[derive(Serialize)]
struct ApiResponse {
    success: bool,
    message: String,
}

#[derive(Deserialize)]
pub struct DetectEmbedRequest {
    pub image: Vec<u8>,
    pub model_path: Option<String>,
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
    let mut redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "detect_embed").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let enrollment = payload.enrollment.unwrap_or(false);

    // 🚀 JUST CALL THE ASYNC FUNCTION DIRECTLY
    let result = detect_and_embed_rust(
        state.clone(),
        payload.image,
        None,
        enrollment
    ).await.map_err(|e| ApiErr {
        status: StatusCode::BAD_REQUEST,
        message: e.to_string(),
    })?;

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
    let mut redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "add_person").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let id = add_face_embedding(
        &payload.school_id,
        payload.embedding,
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
    let mut redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "search_person").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let results = search_person_rust(
        &payload.school_id,
        payload.embedding,
        payload.role,
        payload.k
    ).map_err(|e| ApiErr {
        status: StatusCode::BAD_REQUEST,
        message: e.to_string(),
    })?;

    Ok(Json(SearchPersonResponse { results }))
}

pub async fn verify_attendance_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<VerifyAttendanceRequest>
) -> ApiResult<VerifyAttendanceResponse> {
    let user_key = payload.user_id.to_string();
    let mut redis = state.redis.clone();
    let person_id = payload.user_id.to_string();

    // 1️⃣ Already marked
    if state.attendance.is_recent(&mut redis, &user_key, &person_id).await {
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

    let location_ok = django.is_location_valid(&key).await;

    if !location_ok {
        return Err(ApiErr {
            status: StatusCode::FORBIDDEN,
            message: "User not at required location".into(),
        });
    }

    // 4️⃣ Mark attendance
    let _ = state.attendance.mark(&mut redis, &user_key, &person_id).await;

    Ok(
        Json(VerifyAttendanceResponse {
            status: "marked".into(),
        })
    )
}

// verify_face_handler
pub async fn verify_face_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<VerifyFaceRequest>
) -> ApiResult<VerifyFaceResponse> {
    let redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "verify_face").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let similarity = verify_face_with_pool(
        state.clone(),
        payload.image,
        payload.known_embedding,
        None,
        None
    ).await.map_err(|e| ApiErr {
        status: StatusCode::BAD_REQUEST,
        message: e.to_string(),
    })?;

    Ok(Json(VerifyFaceResponse { similarity }))
}

// detect_emotion_handler
pub async fn detect_emotion_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DetectEmotionRequest>
) -> ApiResult<DetectEmotionResponse> {
    let mut redis = state.redis.clone();

    if !state.security.allow_request(&mut redis, "detect_emotion").await {
        return Err(ApiErr {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "Rate limit exceeded".into(),
        });
    }

    let emotion = detect_emotion_with_pool(state.clone(), payload.image, None, None).await.map_err(
        |e| ApiErr {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        }
    )?;

    Ok(Json(DetectEmotionResponse { emotion }))
}

/// POST /face/save
pub async fn save_face_db_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SaveDbRequest>,
) -> impl IntoResponse {

    let local_dir = std::env::var("FACE_DB_PATH")
        .unwrap_or_else(|_| "face_database".to_string());

    match state.face_db_backup.save(&payload.school_id, &local_dir).await {
        Ok(_) => (
            StatusCode::OK,
            Json(ApiResponse {
                success: true,
                message: "Face DB saved successfully".to_string(),
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Failed: {}", e),
            }),
        ),
    }
}

/// POST /face/init
pub async fn init_face_db_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<InitDbRequest>,
) -> impl IntoResponse {

    let local_dir = std::env::var("FACE_DB_PATH")
        .unwrap_or_else(|_| "face_database".to_string());

    if let Err(e) = std::fs::create_dir_all(&local_dir) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Dir error: {e}"),
            }),
        );
    }

    if let Err(e) = crate::face_db::init_database_rust(&local_dir) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Init failed: {e}"),
            }),
        );
    }

    if let Err(e) = state.face_db_backup
        .save(&payload.school_id, &local_dir)
        .await
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Backup failed: {e}"),
            }),
        );
    }

    (
        StatusCode::OK,
        Json(ApiResponse {
            success: true,
            message: "Initialized".to_string(),
        }),
    )
}

/// POST /face/load
pub async fn load_face_db_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LoadDbRequest>,
) -> impl IntoResponse {

    let local_dir = std::env::var("FACE_DB_PATH")
        .unwrap_or_else(|_| "face_database".to_string());

    match state.face_db_backup.load(&payload.school_id, &local_dir).await {
        Ok(_) => (
            StatusCode::OK,
            Json(ApiResponse {
                success: true,
                message: "Face DB loaded successfully".to_string(),
            }),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("Failed: {}", e),
            }),
        ),
    }
}

pub async fn inference_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.face_pool.is_ready() {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "warming")
    }
}

// ── Routes ──────────────────────────────────────────────────────────────

pub fn face_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/face/detect-embed", post(detect_and_embed_handler))
        .route("/face/add-person", post(detect_and_add_person_handler))
        .route("/face/search", post(search_person_handler))
        .route("/face/verify", post(verify_face_handler))
        .route("/face/emotion", post(detect_emotion_handler))
        .route("/attendance/verify", post(verify_attendance_handler))
        .route("/face/load", post(load_face_db_handler))
        .route("/face/save", post(save_face_db_handler))
        .route("/face/init", post(init_face_db_handler))
        .route("/health/inference", get(inference_health))
}
