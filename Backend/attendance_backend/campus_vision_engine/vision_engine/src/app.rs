/// app.rs

use std::sync::Arc;
use dashmap::DashMap;
use redis::{ Client, aio::MultiplexedConnection };

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
    pub scheduler: PriorityScheduler,
}

impl AppState {
    pub async fn new() -> Self {
        // ── Smart .env loading ─────────────────────────────────────────────────────
        let env_path = if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
            // Best when running via `cargo run` (most common case)
            std::path::PathBuf::from(manifest_dir).join(".env")
        } else {
            // Fallback when running the compiled .exe directly
            std::env::current_dir().unwrap_or_default().join("vision_engine").join(".env")
        };

        if env_path.exists() {
            match dotenvy::from_path_override(&env_path) {
                Ok(_) => println!("✅ Loaded .env from: {}", env_path.display()),
                Err(e) => println!("⚠️  Failed to load .env from {}: {}", env_path.display(), e),
            }
        } else {
            println!("⚠️  .env file NOT found at: {}", env_path.display());
            println!(
                "   Current working dir: {}",
                std::env::current_dir().unwrap_or_default().display()
            );
        }

        // ── Print all relevant env vars for diagnosis ─────────────────────
        Self::diagnose_env();

        // ── Redis ─────────────────────────────────────────────────────────
        let redis = Self::retry_async(
            || async {
                let redis_url = std::env
                    ::var("REDIS_URL")
                    .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

                println!("🔌 Connecting to Redis: {}", redis_url);

                let client = Client::open(redis_url.clone()).map_err(|e|
                    anyhow::anyhow!("Redis client error: {}", e)
                )?;

                let conn = client
                    .get_multiplexed_async_connection().await
                    .map_err(|e|
                        anyhow::anyhow!("Redis connection failed to {}: {}", redis_url, e)
                    )?;

                println!("✅ Redis connected");
                Ok(conn)
            },
            5,
            300
        ).await.unwrap_or_else(|e| {
            panic!("\n\
                ═══════════════════════════════════════════════\n\
                ❌ REDIS CONNECTION FAILED\n\
                Error: {}\n\
                Fix: set REDIS_URL in your .env file\n\
                Example: REDIS_URL=redis://127.0.0.1:6379\n\
                ═══════════════════════════════════════════════", e)
        });

        // ── MinIO ─────────────────────────────────────────────────────────
        println!("🔌 Connecting to MinIO...");

        let face_db_backup = match Self::retry_async(
            || async {
                MinioFaceDb::new().await
                    .map_err(|e| anyhow::anyhow!("MinIO init failed: {:#}", e))  // better formatting
            },
            5,
            1000,  // slightly longer delay for MinIO
        ).await {
            Ok(db) => {
                println!("✅ MinIO client created successfully for bucket: {}", db.bucket);
                db
            }
            Err(e) => {
                eprintln!("\nFull MinIO error: {:#?}", e);  // detailed debug
                panic!(
                    "\n\
                    ═══════════════════════════════════════════════\n\
                    ❌ MINIO FAILED TO INITIALIZE\n\
                    Error: {}\n\n\
                    Quick checks:\n\
                     • Is MinIO running?   →  docker ps\n\
                     • Open browser:       http://localhost:9000  (login: admin / Anshu_hdisd12345)\n\
                     • Bucket exists?      → Create 'face-db-backups' in MinIO console\n\
                     • docker-compose up -d (run from vision_engine folder)\n\
                    ═══════════════════════════════════════════════",
                    e
                );
            }
        };

        // ── Django ────────────────────────────────────────────────────────
        let django_base_url = std::env::var("DJANGO_BASE_URL").unwrap_or_else(|_| {
            println!("⚠️  DJANGO_BASE_URL not set, using http://127.0.0.1:8000");
            "http://127.0.0.1:8000".to_string()
        });
        let django = DjangoService::new(django_base_url);

        // ── Model pools ───────────────────────────────────────────────────
        let cores = num_cpus::get_physical();
        println!("💻 Physical cores: {}", cores);

        let yunet_workers = if cores <= 2 { 1 } else if cores <= 6 { 2 } else { 3 };
        let face_workers = if cores <= 2 { 1 } else if cores <= 6 { 2 } else { 2 };

        println!("🧠 YuNet workers: {}, ArcFace workers: {}", yunet_workers, face_workers);

        // Get absolute path to models folder (this is the key fix)
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map(|p| std::path::PathBuf::from(p))
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default());

        let models_dir = manifest_dir.join("models");

        // Verify models exist
        Self::check_model_file_with_base(&models_dir, "face_detection_yunet_2023mar.onnx");
        Self::check_model_file_with_base(&models_dir, "arcface.onnx");
        Self::check_model_file_with_base(&models_dir, "emotion-ferplus-8.onnx");

        // === CREATE POOLS WITH FULL ABSOLUTE PATHS ===
        let yunet_path = models_dir.join("face_detection_yunet_2023mar.onnx");
        let arcface_path = models_dir.join("arcface.onnx");
        let emotion_path = models_dir.join("emotion-ferplus-8.onnx");

        println!("Loading YuNet from: {}", yunet_path.display());
        let yunet_pool = Arc::new(
            YuNetPool::new(yunet_path.to_str().unwrap(), yunet_workers)
        );
        println!("✅ YuNet pool ready");

        println!("Loading ArcFace from: {}", arcface_path.display());
        let face_pool = Arc::new(
            InferencePool::new(arcface_path.to_str().unwrap(), face_workers)
        );

        println!("Loading Emotion from: {}", emotion_path.display());
        let emotion_pool = Arc::new(
            InferencePool::new(emotion_path.to_str().unwrap(), 1)
        );

        println!("✅ Inference pools created, warming up...");

        // Warmup with correct shapes
        face_pool.warm_up(ndarray::Array4::<f32>::zeros((1, 112, 112, 3)));
        emotion_pool.warm_up(ndarray::Array4::<f32>::zeros((1, 1, 64, 64)));

        println!("✅ Models warmed up successfully");

        println!("✅ AppState ready\n");

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

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn diagnose_env() {
        println!("\n── Environment Check ──────────────────────────────────");
        let vars = [
            "REDIS_URL",
            "MINIO_ENDPOINT",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
            "MINIO_BUCKET_NAME",
            "DJANGO_BASE_URL",
            "FACE_DB_PATH",
            "FACE_DB_ROLE",
            "ATTENDANCE_TTL_SECS",
        ];
        for var in &vars {
            match std::env::var(var) {
                Ok(val) => {
                    if var.contains("KEY") || var.contains("SECRET") {
                        println!("  ✅ {} = [set]", var);
                    } else {
                        println!("  ✅ {} = {}", var, val);
                    }
                }
                Err(_) => println!("  ⚠️  {} = NOT SET (will use default)", var),
            }
        }
        println!("───────────────────────────────────────────────────────\n");
    }

    fn check_model_file_with_base(base_dir: &std::path::Path, filename: &str) {
        let full_path = base_dir.join(filename);
        if full_path.exists() {
            let size = std::fs::metadata(&full_path)
                .map(|m| m.len())
                .unwrap_or(0);
            println!("  ✅ Model: {} ({:.1} MB)", filename, (size as f64) / 1_048_576.0);
        } else {
            panic!(
                "\n\
                ═══════════════════════════════════════════════\n\
                ❌ MODEL FILE NOT FOUND: {}\n\
                Expected at: {}\n\
                Current working dir: {}\n\
                Fix: Make sure the .onnx files are in the models/ folder inside vision_engine/\n\
                ═══════════════════════════════════════════════",
                filename,
                full_path.display(),
                std::env::current_dir().unwrap_or_default().display()
            );
        }
    }

    /// Returns Result so callers can show a useful message instead of
    /// the generic "All retries failed" panic.
    async fn retry_async<F, Fut, T>(mut f: F, retries: usize, delay_ms: u64) -> anyhow::Result<T>
        where F: FnMut() -> Fut, Fut: std::future::Future<Output = anyhow::Result<T>>
    {
        let mut last_err = anyhow::anyhow!("No attempts made");
        for attempt in 1..=retries {
            match f().await {
                Ok(val) => {
                    return Ok(val);
                }
                Err(e) => {
                    println!("  ⚠️  Attempt {}/{} failed: {}", attempt, retries, e);
                    last_err = e;
                    if attempt < retries {
                        tokio::time::sleep(
                            std::time::Duration::from_millis(delay_ms * (attempt as u64))
                        ).await;
                    }
                }
            }
        }
        Err(last_err)
    }
}
