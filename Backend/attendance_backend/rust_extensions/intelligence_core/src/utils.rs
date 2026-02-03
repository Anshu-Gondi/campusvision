/// Simple rectangle type (no OpenCV needed)
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub fn area(&self) -> i32 {
        self.width * self.height
    }
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a_sq = a.iter().map(|x| x * x).sum::<f32>();
    let norm_b_sq = b.iter().map(|x| x * x).sum::<f32>();
    let denom = (norm_a_sq.sqrt() * norm_b_sq.sqrt()).max(1e-8);
    dot / denom
}

/// Compute IoU between two rectangles
pub fn iou(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }
    let inter = (x2 - x1) * (y2 - y1);
    let union = a.area() + b.area() - inter;
    inter as f32 / union as f32
}
