use redis::{ Client, aio::MultiplexedConnection };
use tokio::sync::{ RwLock, Mutex };
use std::{ collections::HashMap, sync::Arc };

use crate::security::SecurityService;
use crate::attendance::AttendanceService;
use crate::django::DjangoService;

#[derive(Clone)]
pub struct AppState {
    /// Shared Redis connection (safe for concurrent handlers)
    pub redis: Arc<Mutex<MultiplexedConnection>>,

    /// session_id → last embedding
    pub last_embeddings: Arc<RwLock<HashMap<String, Vec<f32>>>>,

    pub security: SecurityService,
    pub attendance: AttendanceService,
    pub django: DjangoService,
}

impl AppState {
    pub async fn new() -> Self {
        // Load .env file (only affects dev; safe to call always)
        // .ok() ignores error if .env doesn't exist (e.g. in production)
        dotenvy::dotenv().ok();

        // Redis URL: from .env or fallback (never hardcode in prod!)
        let redis_url = std::env::var("REDIS_URL")
            .expect("REDIS_URL must be set in .env or environment variables");

        let client = Client::open(redis_url)
            .expect("❌ Failed to create Redis client");

        let redis = client
            .get_multiplexed_async_connection()
            .await
            .expect("❌ Failed to connect to Redis");

        // Django base URL: from .env or fallback
        let django_base_url = std::env::var("DJANGO_BASE_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());

        Self {
            redis: Arc::new(Mutex::new(redis)),
            last_embeddings: Arc::new(RwLock::new(HashMap::new())),
            security: SecurityService::new(),
            attendance: AttendanceService::new(),
            django: DjangoService::new(django_base_url),
        }
    }
}