use redis::AsyncCommands;
use anyhow::Result;

#[derive(Clone)]
pub struct AttendanceService {
    ttl_secs: u64,
}

impl AttendanceService {
    pub fn new() -> Self {
        let ttl = std::env::var("ATTENDANCE_TTL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1800); // 30 min default
        Self { ttl_secs: ttl }
    }

    fn key(branch_id: &str, person_id: &str) -> String {
        format!("attendance:{}:{}", branch_id, person_id)
    }

    pub async fn is_recent(
        &self,
        redis: &mut redis::aio::MultiplexedConnection,
        branch_id: &str,
        person_id: &str,
    ) -> bool {
        match redis.exists(Self::key(branch_id, person_id)).await {
            Ok(exists) => exists,
            Err(e) => {
                log::warn!("Redis check failed for {}:{} — {}", branch_id, person_id, e);
                false // fail open
            }
        }
    }

    pub async fn mark(
        &self,
        redis: &mut redis::aio::MultiplexedConnection,
        branch_id: &str,
        person_id: &str,
    ) -> Result<()> {
        redis
            .set_ex(Self::key(branch_id, person_id), "1", self.ttl_secs)
            .await
            .map_err(|e| anyhow::anyhow!("Redis mark failed: {}", e))
    }
}