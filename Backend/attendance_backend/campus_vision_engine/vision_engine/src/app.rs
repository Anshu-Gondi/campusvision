use redis::{ Client, aio::MultiplexedConnection };
use tokio::sync::{ RwLock, Mutex };
use std::{ collections::HashMap, sync::Arc };

use crate::service::security::SecurityService;
use crate::service::attendance::AttendanceService;
use crate::service::django::DjangoService;
use crate::storage::minio_face_db::MinioFaceDb;
use crate::models::onnx_models::InferencePool;
use crate::models::yunet_pool::YuNetPool;
#[derive(Clone)]
pub struct AppState {
    pub redis: Arc<Mutex<MultiplexedConnection>>,
    pub last_embeddings: Arc<RwLock<HashMap<String, Vec<f32>>>>,

    // 🔥 NEW: inference pools
    pub yunet_pool: Arc<YuNetPool>,
    pub face_pool: Arc<InferencePool>,
    pub emotion_pool: Arc<InferencePool>,

    pub security: SecurityService,
    pub attendance: AttendanceService,
    pub django: DjangoService,

    pub face_db_backup: MinioFaceDb,
}

pub async fn new() -> Self {
    dotenvy::dotenv().ok();

    let startup = std::time::Instant::now();

    // ── Redis ─────────────────────────────
    let redis_url =
        std::env::var("REDIS_URL").expect("REDIS_URL must be set");

    let client = Client::open(redis_url)
        .expect("Failed to create Redis client");

    let redis = client
        .get_multiplexed_async_connection()
        .await
        .expect("Failed to connect to Redis");

    // ── Django ────────────────────────────
    let django_base_url = std::env::var("DJANGO_BASE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());

    // ── MinIO ─────────────────────────────
    let face_db_backup =
        MinioFaceDb::new()
            .await
            .expect("Failed to init MinIO");

    let cpu_count = num_cpus::get();

    let yunet_workers = if cpu_count >= 8 { 3 }
        else if cpu_count >= 4 { 2 }
        else { 1 };

    let face_workers = yunet_workers;
    let emotion_workers = 1;

    // 🔥 Create pools
    let yunet_pool = Arc::new(
        YuNetPool::new(
            "models/face_detection_yunet_2023mar.onnx",
            yunet_workers
        )
    );

    let face_pool = Arc::new(
        InferencePool::new("models/facenet.onnx", face_workers)
    );

    let emotion_pool = Arc::new(
        InferencePool::new("models/emotion.onnx", emotion_workers)
    );

    // 🔥 Warmup
    println!("Warming up models...");

    let dummy = ndarray::Array4::<f32>::zeros((1, 3, 112, 112));
    face_pool.warm_up(dummy.clone());
    emotion_pool.warm_up(dummy);

    println!(
        "Startup completed in {:.2?}",
        startup.elapsed()
    );

    Self {
        redis: Arc::new(Mutex::new(redis)),
        last_embeddings: Arc::new(RwLock::new(HashMap::new())),

        yunet_pool,
        face_pool,
        emotion_pool,

        security: SecurityService::new(),
        attendance: AttendanceService::new(),
        django: DjangoService::new(django_base_url),
        face_db_backup,
    }
}
