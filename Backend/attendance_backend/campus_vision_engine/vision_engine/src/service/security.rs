use redis::{AsyncCommands, Script};
use anyhow::Result;

#[derive(Clone)]
pub struct SecurityService {
    max_requests: u32,
    window_secs: i64,
    max_face_attempts: u32,
}

impl SecurityService {
    pub fn new() -> Self {
        Self {
            max_requests: 10,
            window_secs: 10,
            max_face_attempts: 5,
        }
    }

    async fn run_lua(&self, redis: &mut redis::aio::MultiplexedConnection, key: &str, window: i64) -> u32 {
        let script = Script::new(r#"
            local count = redis.call('INCR', KEYS[1])
            if count == 1 then
                redis.call('EXPIRE', KEYS[1], ARGV[1])
            end
            return count
        "#);

        script.key(key).arg(window).invoke_async(redis).await.unwrap_or(0)
    }

    pub async fn allow_request(&self, redis: &mut redis::aio::MultiplexedConnection, key: &str) -> bool {
        let redis_key = format!("rate:{}", key);
        self.run_lua(redis, &redis_key, self.window_secs).await <= self.max_requests
    }

    pub async fn allow_camera_frame(&self, redis: &mut redis::aio::MultiplexedConnection, camera_id: &str) -> bool {
        let key = format!("rate:cam:{}", camera_id);
        self.run_lua(redis, &key, 1).await <= 30
    }

    pub async fn validate_attempt(&self, redis: &mut redis::aio::MultiplexedConnection, user_id: &str) -> Result<(), String> {
        let key = format!("attempt:{}", user_id);
        let attempts = self.run_lua(redis, &key, 60).await;

        if attempts > self.max_face_attempts {
            return Err("Too many verification attempts. Try again in 60 seconds.".to_string());
        }

        Ok(())
    }
}