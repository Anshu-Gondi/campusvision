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
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
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

    // ✅ floor each norm separately
    let denom = norm_a_sq.sqrt().max(1e-8) * norm_b_sq.sqrt().max(1e-8);
    dot / denom
}

#[inline]
pub fn normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(1e-8);
    for x in v {
        *x /= norm;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn cosine_similarity_sse41(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;

    let mut vdot    = _mm_setzero_ps();
    let mut vnorm_a = _mm_setzero_ps();
    let mut vnorm_b = _mm_setzero_ps();

    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        vdot    = _mm_add_ps(vdot,    _mm_mul_ps(va, vb));
        vnorm_a = _mm_add_ps(vnorm_a, _mm_mul_ps(va, va));
        vnorm_b = _mm_add_ps(vnorm_b, _mm_mul_ps(vb, vb));
        i += 4;
    }

    let mut dot    = hsum_sse(vdot);
    let mut norm_a = hsum_sse(vnorm_a);
    let mut norm_b = hsum_sse(vnorm_b);

    while i < len {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    // ✅ floor each norm BEFORE multiplying — not the product
    let denom = norm_a.sqrt().max(1e-8) * norm_b.sqrt().max(1e-8);
    dot / denom
}

// horizontal sum of 4 f32 lanes into one f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn hsum_sse(v: __m128) -> f32 {
    // _mm_hadd_ps adds adjacent pairs: [a+b, c+d, a+b, c+d]
    let shuf = _mm_hadd_ps(v, v);
    // second hadd: [(a+b)+(c+d), ...] = full sum in lane 0
    let sums = _mm_hadd_ps(shuf, shuf);
    _mm_cvtss_f32(sums)
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

// test helpers
pub fn scalar_norm(a: &[f32]) -> f32 {
    a.iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
}

pub fn make_vec(seed: f32, len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * seed + 0.1).sin()).collect()
}
