/// app.rs
///
/// Key changes vs original:
///   1. `PriorityScheduler` added — shared across all routes.
///   2. `school_id` context added to face_db_backup operations.
///   3. Redis clone pattern documented — MultiplexedConnection is
///      already cheaply clone-able; no pool needed.

use anyhow::Result;
use std::sync::Arc;
use dashmap::DashMap;
use redis::{Client, aio::MultiplexedConnection};

use crate::service::attendance::AttendanceService;
use crate::service::django::DjangoService;
use crate::service::security::SecurityService;
use crate::storage::minio_face_db::MinioFaceDb;
use crate::models::onnx_models::InferencePool;
use crate::models::yunet_pool::YuNetPool;
use crate::scheduler::PriorityScheduler;

#[derive(Clone)]
pub struct AppState {
    pub redis: MultiplexedConnection,
    pub last_embeddings: Arc<DashMap<String, Vec<f32>>>,
    pub yunet_pool: Arc<YuNetPool>,
    pub face_pool: Arc<InferencePool>,
    pub emotion_pool: Arc<InferencePool>,
    pub security: SecurityService,
    pub attendance: AttendanceService,
    pub django: DjangoService,
    pub face_db_backup: MinioFaceDb,

    /// Priority scheduler — direct lane vs CCTV lane.
    /// All route handlers get this via `State<Arc<AppState>>`.
    pub scheduler: PriorityScheduler,
}

impl AppState {
    pub async fn new() -> Self {
        dotenvy::dotenv().ok();

        let redis = Self::retry_async(
            || async {
                let redis_url = std::env::var("REDIS_URL").map_err(|e| anyhow::anyhow!(e))?;
                let client = Client::open(redis_url).map_err(|e| anyhow::anyhow!(e))?;
                let conn = client
                    .get_multiplexed_async_connection()
                    .await
                    .map_err(|e| anyhow::anyhow!(e))?;
                Ok(conn)
            },
            5,
            300,
        )
        .await;

        let face_db_backup = Self::retry_async(
            || async { MinioFaceDb::new().await.map_err(|e| anyhow::anyhow!(e)) },
            5,
            500,
        )
        .await;

        let django_base_url = std::env::var("DJANGO_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
        let django = DjangoService::new(django_base_url);

        let cores = num_cpus::get_physical();

        // Pool sizes: tuned for cloud auto-scaling (≥2 cores assumed).
        // YuNet is lightweight; ArcFace is the expensive one.
        let yunet_workers = if cores <= 2 { 1 } else if cores <= 6 { 2 } else { 3 };
        let face_workers  = if cores <= 2 { 1 } else if cores <= 6 { 2 } else { 2 };

        let yunet_pool = Arc::new(
            YuNetPool::new("models/face_detection_yunet_2023mar.onnx", yunet_workers),
        );
        let face_pool = Arc::new(InferencePool::new("models/arcface.onnx", face_workers));
        let emotion_pool = Arc::new(InferencePool::new("models/emotion.onnx", 1));

        // Warm up inference workers so first real requests aren't slow.
        let dummy = ndarray::Array4::<f32>::zeros((1, 3, 112, 112));
        face_pool.warm_up(dummy.clone());
        emotion_pool.warm_up(dummy);

        Self {
            redis,
            last_embeddings: Arc::new(DashMap::new()),
            yunet_pool,
            face_pool,
            emotion_pool,
            security: SecurityService::new(),
            attendance: AttendanceService::new(),
            django,
            face_db_backup,
            scheduler: PriorityScheduler::new(),
        }
    }

    async fn retry_async<F, Fut, T>(mut f: F, retries: usize, delay_ms: u64) -> T
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = anyhow::Result<T>>,
    {
        let mut last_err = None;
        for i in 0..retries {
            match f().await {
                Ok(val) => return val,
                Err(e) => {
                    last_err = Some(e);
                    tokio::time::sleep(std::time::Duration::from_millis(
                        delay_ms * ((i as u64) + 1),
                    ))
                    .await;
                }
            }
        }
        panic!("All retries failed: {:?}", last_err);
    }
}