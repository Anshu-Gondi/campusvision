use std::sync::Arc;
use opencv::core::{Rect, Point2f};
use std::time::Instant;

#[derive(Clone)]
pub struct TrackedFace {
    pub track_id: usize,
    pub person_id: Option<usize>,
    pub embedding: Arc<Vec<f32>>,
    pub emotion: Option<i64>,
    pub bbox: Rect,
    pub landmarks: Vec<Point2f>,
    pub hits: u32,
    pub age: u32,
    pub last_seen: Instant,
    pub confidence: f32,
    pub id_locked: bool,
    pub last_embedding_update: u32,
}
