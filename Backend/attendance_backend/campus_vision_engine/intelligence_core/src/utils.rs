#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

/// Public cosine similarity (auto SIMD if available)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            unsafe {
                return cosine_similarity_sse41(a, b);
            }
        }
    }

    cosine_similarity_scalar(a, b)
}

#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>();
    let norm_a_sq = a
        .iter()
        .map(|x| x * x)
        .sum::<f32>();
    let norm_b_sq = b
        .iter()
        .map(|x| x * x)
        .sum::<f32>();
    let denom = (norm_a_sq.sqrt() * norm_b_sq.sqrt()).max(1e-8);
    dot / denom
}

#[inline]
pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v {
        *x /= norm;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn cosine_similarity_sse41(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut i = 0;

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    // process 4 floats at a time
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));

        // dot product (all lanes)
        let dp = _mm_dp_ps(va, vb, 0xFF);
        dot += _mm_cvtss_f32(dp);

        // norms
        let na = _mm_dp_ps(va, va, 0xFF);
        let nb = _mm_dp_ps(vb, vb, 0xFF);

        norm_a += _mm_cvtss_f32(na);
        norm_b += _mm_cvtss_f32(nb);

        i += 4;
    }

    // tail (remaining elements)
    while i < len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
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

    (inter as f32) / (union as f32)
}
