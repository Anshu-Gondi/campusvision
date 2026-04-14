#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

// ── Public API ────────────────────────────────────────────────────────────────
//
// Runtime dispatch: SSE4.1 → scalar (unrolled).
// AVX/AVX2 intentionally excluded — not present on all target hardware.
// Adding AVX later is safe: just add a tier above SSE4.1.

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            // SAFETY: guarded by runtime feature detection above.
            return unsafe { cosine_similarity_sse41(a, b) };
        }
    }

    cosine_similarity_scalar(a, b)
}

pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

// ── Scalar path — unrolled 8-wide ────────────────────────────────────────────
//
// The compiler will auto-vectorize this with SSE2 (baseline x86_64)
// even without explicit intrinsics, so this is not "slow scalar" —
// it's safe-portable-vectorizable code.

#[inline]
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    // 8-wide unroll — compiler will SSE2-vectorize this automatically
    for i in 0..chunks {
        let base = i * 8;
        dot    += a[base]   * b[base]
                + a[base+1] * b[base+1]
                + a[base+2] * b[base+2]
                + a[base+3] * b[base+3]
                + a[base+4] * b[base+4]
                + a[base+5] * b[base+5]
                + a[base+6] * b[base+6]
                + a[base+7] * b[base+7];

        norm_a += a[base]   * a[base]
                + a[base+1] * a[base+1]
                + a[base+2] * a[base+2]
                + a[base+3] * a[base+3]
                + a[base+4] * a[base+4]
                + a[base+5] * a[base+5]
                + a[base+6] * a[base+6]
                + a[base+7] * a[base+7];

        norm_b += b[base]   * b[base]
                + b[base+1] * b[base+1]
                + b[base+2] * b[base+2]
                + b[base+3] * b[base+3]
                + b[base+4] * b[base+4]
                + b[base+5] * b[base+5]
                + b[base+6] * b[base+6]
                + b[base+7] * b[base+7];
    }

    // Remainder
    let base = chunks * 8;
    for i in 0..remainder {
        dot    += a[base+i] * b[base+i];
        norm_a += a[base+i] * a[base+i];
        norm_b += b[base+i] * b[base+i];
    }

    let denom = norm_a.sqrt().max(1e-8) * norm_b.sqrt().max(1e-8);
    dot / denom
}

// ── SSE4.1 explicit path ──────────────────────────────────────────────────────
//
// Only compiled on x86_64. Only called after runtime feature check.
// Uses 4-wide f32 SIMD (128-bit XMM registers) — present on all
// CPUs from ~2007 onwards including yours (SSE4.2 ⊃ SSE4.1).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn cosine_similarity_sse41(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;

        let mut vdot    = _mm_setzero_ps();
        let mut vnorm_a = _mm_setzero_ps();
        let mut vnorm_b = _mm_setzero_ps();

        for i in 0..chunks {
            let offset = i * 4;
            let va = _mm_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm_loadu_ps(b.as_ptr().add(offset));
            vdot    = _mm_add_ps(vdot,    _mm_mul_ps(va, vb));
            vnorm_a = _mm_add_ps(vnorm_a, _mm_mul_ps(va, va));
            vnorm_b = _mm_add_ps(vnorm_b, _mm_mul_ps(vb, vb));
        }

        let mut dot    = hsum_sse(vdot);
        let mut norm_a = hsum_sse(vnorm_a);
        let mut norm_b = hsum_sse(vnorm_b);

        // Remainder
        let base = chunks * 4;
        for i in 0..remainder {
            dot    += a[base+i] * b[base+i];
            norm_a += a[base+i] * a[base+i];
            norm_b += b[base+i] * b[base+i];
        }

        let denom = norm_a.sqrt().max(1e-8) * norm_b.sqrt().max(1e-8);
        dot / denom
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn hsum_sse(v: __m128) -> f32 {
    unsafe {
        let shuf = _mm_hadd_ps(v, v);
        let sums = _mm_hadd_ps(shuf, shuf);
        _mm_cvtss_f32(sums)
    }
}

// ── Geometry ──────────────────────────────────────────────────────────────────

pub fn iou(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);

    if x2 <= x1 || y2 <= y1 { return 0.0; }

    let inter = (x2 - x1) * (y2 - y1);
    let union = a.area() + b.area() - inter;
    (inter as f32) / (union as f32)
}

pub fn scalar_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn make_vec(seed: f32, len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * seed + 0.1).sin()).collect()
}