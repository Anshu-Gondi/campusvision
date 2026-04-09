use anyhow::Result;
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

#[derive(Clone)]
pub struct DjangoService {
    base_url: String,
    client: Client,
}

impl DjangoService {
    pub fn new(base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(2)) // slightly safer than 500ms
            .build()
            .expect("Failed to build HTTP client");
        Self { base_url, client }
    }

    pub async fn is_location_valid(&self, user_id: &str) -> bool {
        let url = format!("{}/api/attendance/location-check/", self.base_url);
        match self.client
            .post(&url)
            .json(&serde_json::json!({ "user_id": user_id }))
            .send()
            .await
        {
            Ok(res) => match res.json::<Value>().await {
                Ok(json) => json["valid"].as_bool().unwrap_or(false),
                Err(e) => {
                    log::warn!("Failed parsing JSON from Django: {}", e);
                    false
                }
            },
            Err(e) => {
                log::warn!("Django location check failed: {}", e);
                false
            }
        }
    }

    pub async fn notify_attendance(
        &self,
        branch_id: &str,
        person_id: &str,
        role: &str,
        method: &str,
    ) -> Result<()> {
        let url = format!("{}/api/attendance/mark/", self.base_url);
        self.client
            .post(&url)
            .json(&serde_json::json!({
                "branch_id": branch_id,
                "person_id": person_id,
                "role": role,
                "method": method,
            }))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("notify_attendance failed: {}", e))?;
        Ok(())
    }

    pub async fn get_enrolled_ids(&self, branch_id: &str) -> Result<Vec<String>> {
        let url = format!("{}/api/faces/enrolled/?branch_id={}", self.base_url, branch_id);
        let res = self.client.get(&url).send().await?;
        Ok(res.json::<Vec<String>>().await?)
    }
}