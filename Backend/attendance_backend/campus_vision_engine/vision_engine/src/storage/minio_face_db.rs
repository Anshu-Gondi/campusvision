use aws_sdk_s3::{Client, primitives::ByteStream};
use aws_config::{
    meta::region::RegionProviderChain,
    BehaviorVersion,
};
use anyhow::{Result, anyhow};
use tokio::fs;
use std::path::Path;

use crate::face_db;

#[derive(Clone)]
pub struct MinioFaceDb {
    client: Client,
    bucket: String,
}

impl MinioFaceDb {
    pub async fn new() -> Result<Self> {
        let endpoint = std::env::var("MINIO_ENDPOINT")
            .expect("MINIO_ENDPOINT missing");

        let access_key = std::env::var("MINIO_ACCESS_KEY")
            .expect("MINIO_ACCESS_KEY missing");

        let secret_key = std::env::var("MINIO_SECRET_KEY")
            .expect("MINIO_SECRET_KEY missing");

        let bucket = std::env::var("MINIO_BUCKET_NAME")
            .expect("MINIO_BUCKET_NAME missing");

        let region_provider =
            RegionProviderChain::default_provider().or_else("us-east-1");

        let credentials = aws_sdk_s3::config::Credentials::new(
            access_key,
            secret_key,
            None,
            None,
            "minio",
        );

        let config = aws_config::defaults(BehaviorVersion::latest())
            .region(region_provider)
            .endpoint_url(endpoint)
            .credentials_provider(credentials)
            .load()
            .await;

        let client = Client::new(&config);

        Ok(Self { client, bucket })
    }

    pub async fn save(&self, local_dir: &str) -> Result<()> {
        face_db::save_database_rust(local_dir)?;

        let path = Path::new(local_dir).join("data.bin");
        let bytes = fs::read(&path).await?;
        let body = ByteStream::from(bytes);

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key("face-db/data.bin")
            .body(body)
            .send()
            .await?;

        Ok(())
    }

    pub async fn load(&self, local_dir: &str) -> Result<()> {
        let obj = self.client
            .get_object()
            .bucket(&self.bucket)
            .key("face-db/data.bin")
            .send()
            .await
            .map_err(|_| anyhow!("Face DB not found in MinIO"))?;

        let data = obj.body.collect().await?.into_bytes();

        fs::create_dir_all(local_dir).await?;
        fs::write(Path::new(local_dir).join("data.bin"), data).await?;

        face_db::load_database_rust(local_dir)?;

        Ok(())
    }
}
