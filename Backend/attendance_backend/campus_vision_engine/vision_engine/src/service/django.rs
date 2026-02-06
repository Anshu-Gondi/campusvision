use ureq;
use serde_json::Value;

#[derive(Clone)]
pub struct DjangoService {
    base_url: String,
}

impl DjangoService {
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    pub fn is_location_valid(&self, user_id: &str) -> bool {
        let url = format!("{}/api/attendance/location-check/", self.base_url);

        let response = ureq::post(&url)
            .set("Content-Type", "application/json")
            .send_json(serde_json::json!({
                "user_id": user_id
            }));

        match response {
            Ok(resp) => {
                // ureq 2.10 .into_json::<T>() works perfectly here
                match resp.into_json::<Value>() {
                    Ok(json) => json["valid"].as_bool().unwrap_or(false),
                    Err(e) => {
                        eprintln!("JSON parse error: {}", e);
                        false
                    }
                }
            }
            Err(e) => {
                eprintln!("Request failed: {}", e);
                false // fail closed — safe default
            }
        }
    }
}