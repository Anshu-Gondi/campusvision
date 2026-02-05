use redis::AsyncCommands;

#[derive(Clone)]
pub struct AttendanceService {
    ttl_secs: usize,
}

impl AttendanceService {
    pub fn new() -> Self {
        Self {
            ttl_secs: 30 * 60, // 30 minutes
        }
    }

    pub async fn is_recent(
        &self,
        redis: &mut redis::aio::MultiplexedConnection,
        user_id: &str,
    ) -> bool {
        let key = format!("attendance:{}", user_id);
        redis.exists(key).await.unwrap_or(false)
    }

    pub async fn mark(
        &self,
        redis: &mut redis::aio::MultiplexedConnection,
        user_id: &str,
    ) {
        let key = format!("attendance:{}", user_id);
        let _: () = redis
            .set_ex(key, "1", self.ttl_secs as u64)
            .await
            .unwrap_or(());
    }
}
