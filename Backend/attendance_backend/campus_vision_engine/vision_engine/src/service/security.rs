use redis::AsyncCommands;

#[derive(Clone)]
pub struct SecurityService {
    max_requests: u32,
    window_secs: usize,
}

impl SecurityService {
    pub fn new() -> Self {
        Self {
            max_requests: 10,
            window_secs: 10, // 10 sec window
        }
    }

    /// Generic rate limit (API abuse)
    pub async fn allow_request(
        &self,
        redis: &mut redis::aio::MultiplexedConnection,
        key: &str,
    ) -> bool {
        let redis_key = format!("rate:{}", key);

        let count: u32 = redis
            .incr(&redis_key, 1)
            .await
            .unwrap_or(self.max_requests + 1);

        if count == 1 {
            let _: () = redis
                .expire(&redis_key, self.window_secs.try_into().unwrap())
                .await
                .unwrap_or(());
        }

        count <= self.max_requests
    }

    /// Prevent brute-force face attempts
    pub async fn validate_attempt(
        &self,
        redis: &mut redis::aio::MultiplexedConnection,
        user_id: &str,
    ) -> Result<(), String> {
        let key = format!("attempt:{}", user_id);
        let attempts: u32 = redis.incr(&key, 1).await.unwrap_or(5);

        if attempts == 1 {
            let _: () = redis.expire(&key, 60).await.unwrap_or(()); // 1 min
        }

        if attempts > 5 {
            return Err("Too many verification attempts".into());
        }

        Ok(())
    }
}
