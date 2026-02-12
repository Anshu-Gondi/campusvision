use redis::{Client, aio::MultiplexedConnection};
use tokio::sync::{RwLock, Mutex, Semaphore};
use std::{collections::HashMap, sync::Arc};

use crate::service::security::SecurityService;
use crate::service::attendance::AttendanceService;
use crate::service::django::DjangoService;
use crate::storage::minio_face_db::MinioFaceDb;

#[derive(Clone)]
pub struct AppState {
    pub redis: Arc<Mutex<MultiplexedConnection>>,
    pub last_embeddings: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    // 🔒 HARD LIMIT: 1 face inference at a time
    pub face_infer_sem: Arc<Semaphore>,

    pub security: SecurityService,
    pub attendance: AttendanceService,
    pub django: DjangoService,

    pub face_db_backup: MinioFaceDb,
}

impl AppState {
    pub async fn new() -> Self {
        // Load env (dev-safe)
        dotenvy::dotenv().ok();

        // ── Redis ─────────────────────────────
        let redis_url = std::env::var("REDIS_URL")
            .expect("REDIS_URL must be set");

        let client = Client::open(redis_url)
            .expect("❌ Failed to create Redis client");

        let redis = client
            .get_multiplexed_async_connection()
            .await
            .expect("❌ Failed to connect to Redis");

        // ── Django ────────────────────────────
        let django_base_url = std::env::var("DJANGO_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());

        // ── MinIO ─────────────────────────────
        let face_db_backup = MinioFaceDb::new()
            .await
            .expect("❌ Failed to initialize MinIO face DB");

        Self {
            redis: Arc::new(Mutex::new(redis)),
            last_embeddings: Arc::new(RwLock::new(HashMap::new())),
            face_infer_sem: Arc::new(Semaphore::new(1)),
            security: SecurityService::new(),
            attendance: AttendanceService::new(),
            django: DjangoService::new(django_base_url),
            face_db_backup,
        }
    }
}
