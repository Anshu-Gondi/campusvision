use aws_sdk_s3::{Client, primitives::ByteStream};
use aws_config::{meta::region::RegionProviderChain, BehaviorVersion};
use anyhow::{Result, anyhow};
use tokio::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::face_db;

#[derive(Clone)]
pub struct MinioFaceDb {
    client: Client,
    bucket: String,
    lock: Arc<Mutex<()>>,
}

impl MinioFaceDb {
    pub async fn new() -> Result<Self> {
        let endpoint = std::env::var("MINIO_ENDPOINT")?;
        let access_key = std::env::var("MINIO_ACCESS_KEY")?;
        let secret_key = std::env::var("MINIO_SECRET_KEY")?;
        let bucket = std::env::var("MINIO_BUCKET_NAME")?;

        let region_provider = RegionProviderChain::default_provider().or_else("us-east-1");

        let credentials = aws_sdk_s3::config::Credentials::new(
            access_key,
            secret_key,
            None,
            None,
            "minio"
        );

        let config = aws_config::defaults(BehaviorVersion::latest())
            .region(region_provider)
            .endpoint_url(endpoint)
            .credentials_provider(credentials)
            .load()
            .await;

        let s3_config = aws_sdk_s3::config::Builder::from(&config)
            .force_path_style(true)
            .build();

        let client = Client::from_conf(s3_config);

        Ok(Self {
            client,
            bucket,
            lock: Arc::new(Mutex::new(())),
        })
    }

    // ==========================
    // 🔥 SAVE (MULTI-TENANT + DELTA)
    // ==========================
    pub async fn save(&self, school_id: &str, local_dir: &str) -> Result<()> {
        let _guard = self.lock.lock().await;

        let local_dir = local_dir.to_string();
        let local_dir_for_blocking = local_dir.clone();

        // 🔒 Save locally (atomic)
        tokio::task::spawn_blocking(move || {
            face_db::save_database_rust(&local_dir_for_blocking)
        })
        .await
        .map_err(|e| anyhow!("Join error: {e}"))??;

        let path = Path::new(&local_dir).join("data.bin");
        let bytes = fs::read(&path).await?;

        // 🔥 timestamp for delta
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();

        // ─────────────────────────────
        // 1️⃣ Upload DELTA (frequent)
        // ─────────────────────────────
        let delta_key = format!("face-db/{}/deltas/{}.bin", school_id, ts);

        self.upload_with_retry(&delta_key, &bytes).await?;

        // ─────────────────────────────
        // 2️⃣ Upload SNAPSHOT (rare trigger)
        // ─────────────────────────────
        if ts % 10 == 0 { // ⚠️ simple heuristic (replace later)
            let snapshot_key = format!("face-db/{}/snapshot.bin", school_id);
            self.upload_with_retry(&snapshot_key, &bytes).await?;
        }

        Ok(())
    }

    // ==========================
    // 🔥 LOAD (SNAPSHOT + DELTAS)
    // ==========================
    pub async fn load(&self, school_id: &str, local_dir: &str) -> Result<()> {
        let _guard = self.lock.lock().await;

        let snapshot_key = format!("face-db/{}/snapshot.bin", school_id);

        // 1️⃣ Load snapshot
        let obj = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(&snapshot_key)
            .send()
            .await
            .map_err(|_| anyhow!("Snapshot not found"))?;

        let data = obj.body.collect().await?.into_bytes();

        fs::create_dir_all(local_dir).await?;

        let final_path = Path::new(local_dir).join("data.bin");
        let tmp_path = Path::new(local_dir).join("data.bin.tmp");

        fs::write(&tmp_path, &data).await?;
        fs::rename(&tmp_path, &final_path).await?;

        let local_dir_owned = local_dir.to_string();

        tokio::task::spawn_blocking(move || {
            face_db::load_database_rust(&local_dir_owned)
        })
        .await
        .map_err(|e| anyhow!("Join error: {e}"))??;

        // ⚠️ NOTE:
        // Proper delta replay requires embeddings support
        // Currently skipped

        Ok(())
    }

    // ==========================
    // 🔁 RETRY HELPER
    // ==========================
    async fn upload_with_retry(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let mut last_err = None;

        for _ in 0..3 {
            let body = ByteStream::from(bytes.to_vec());

            match self.client
                .put_object()
                .bucket(&self.bucket)
                .key(key)
                .body(body)
                .send()
                .await
            {
                Ok(_) => return Ok(()),
                Err(e) => last_err = Some(e),
            }
        }

        Err(anyhow!("Upload failed for {}: {:?}", key, last_err))
    }
}