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
        if is_x86_feature_detected!("sse3") {
            unsafe {
                return cosine_similarity_sse(a, b);
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn cosine_similarity_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;

    let mut dot_sum = _mm_setzero_ps();
    let mut norm_a_sum = _mm_setzero_ps();
    let mut norm_b_sum = _mm_setzero_ps();

    while i + 4 <= len {
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));

            dot_sum = _mm_add_ps(dot_sum, _mm_mul_ps(va, vb));
            norm_a_sum = _mm_add_ps(norm_a_sum, _mm_mul_ps(va, va));
            norm_b_sum = _mm_add_ps(norm_b_sum, _mm_mul_ps(vb, vb));
        }

        i += 4;
    }

    let dot = unsafe { horizontal_sum(dot_sum) };
    let norm_a = unsafe { horizontal_sum(norm_a_sum) };
    let norm_b = unsafe { horizontal_sum(norm_b_sum) };

    let mut dot_scalar = 0.0;
    let mut norm_a_scalar = 0.0;
    let mut norm_b_scalar = 0.0;

    while i < len {
        dot_scalar += a[i] * b[i];
        norm_a_scalar += a[i] * a[i];
        norm_b_scalar += b[i] * b[i];
        i += 1;
    }

    let dot_total = dot + dot_scalar;
    let norm_a_total = norm_a + norm_a_scalar;
    let norm_b_total = norm_b + norm_b_scalar;

    let denom = (norm_a_total.sqrt() * norm_b_total.sqrt()).max(1e-8);
    dot_total / denom
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum(v: __m128) -> f32 {
    unsafe {
        let shuf = _mm_movehdup_ps(v);
        let sums = _mm_add_ps(v, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }
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
