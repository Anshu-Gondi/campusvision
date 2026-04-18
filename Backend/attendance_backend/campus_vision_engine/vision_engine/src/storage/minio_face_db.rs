use aws_sdk_s3::{ Client, primitives::ByteStream };
use aws_config::{ meta::region::RegionProviderChain, BehaviorVersion };
use anyhow::{ Result, anyhow };
use tokio::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{ SystemTime, UNIX_EPOCH };
use dashmap::DashMap;

use crate::face_db;

#[derive(Clone)]
pub struct MinioFaceDb {
    client: Client,
    pub bucket: String,
    locks: Arc<DashMap<String, Arc<Mutex<()>>>>,
}

impl MinioFaceDb {
    fn get_lock(&self, school_id: &str) -> Arc<Mutex<()>> {
        self.locks
            .entry(school_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

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

        let config = aws_config
            ::defaults(BehaviorVersion::latest())
            .region(region_provider)
            .endpoint_url(endpoint)
            .credentials_provider(credentials)
            .load().await;

        let s3_config = aws_sdk_s3::config::Builder::from(&config).force_path_style(true).build();

        Ok(Self {
            client: Client::from_conf(s3_config),
            bucket,
            locks: Arc::new(DashMap::new()),
        })
    }

    // 🔥 SAVE
    pub async fn save(&self, school_id: &str, local_dir: &str) -> Result<()> {
        let lock = self.get_lock(school_id);
        let _guard = lock.lock().await;

        // ⚠️ GLOBAL SAVE (you need per-school later)
        tokio::task
            ::spawn_blocking({
                let local_dir = local_dir.to_string();
                move || face_db::save_database_rust(&local_dir)
            }).await
            .map_err(|e| anyhow!("Join error: {e}"))??;

        let path = Path::new(local_dir).join("data.bin");
        let bytes = fs::read(&path).await?;

        let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        let version_key = format!("face-db/{}/snapshots/{}.bin", school_id, ts);
        let latest_key = format!("face-db/{}/latest.bin", school_id);

        self.upload(&version_key, &bytes).await?;
        self.upload(&latest_key, &bytes).await?;

        Ok(())
    }

    // 🔥 LOAD
    pub async fn load(&self, school_id: &str, local_dir: &str) -> Result<()> {
        let lock = self.get_lock(school_id);
        let _guard = lock.lock().await;

        let key = format!("face-db/{}/latest.bin", school_id);

        let obj = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send().await
            .map_err(|_| anyhow!("No snapshot found"))?;

        let data = obj.body.collect().await?.into_bytes();

        fs::create_dir_all(local_dir).await?;

        let tmp = Path::new(local_dir).join("data.bin.tmp");
        let final_path = Path::new(local_dir).join("data.bin");

        fs::write(&tmp, &data).await?;
        fs::rename(&tmp, &final_path).await?;

        tokio::task
            ::spawn_blocking({
                let local_dir = local_dir.to_string();
                move || face_db::load_database_rust(&local_dir)
            }).await
            .map_err(|e| anyhow!("Join error: {e}"))??;

        Ok(())
    }

    async fn upload(&self, key: &str, bytes: &[u8]) -> Result<()> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(ByteStream::from(bytes.to_vec()))
            .send().await
            .map_err(|e| anyhow!("Upload failed {}: {}", key, e))?;

        Ok(())
    }
}
