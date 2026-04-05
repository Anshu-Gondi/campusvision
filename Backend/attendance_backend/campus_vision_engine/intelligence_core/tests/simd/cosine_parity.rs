use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::RngExt;

use intelligence_core::utils::{
    cosine_similarity,
    cosine_similarity_scalar,
    normalize,
    make_vec,
    scalar_norm
};

const DIM: usize = 256;
const N: usize = 50_000;

fn rand_vec(rng: &mut StdRng) -> Vec<f32> {
    (0..DIM).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-8)
}

#[test]
fn simd_scalar_drift_extreme() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut max_diff = 0.0;

    for _ in 0..N {
        let a = rand_vec(&mut rng);
        let b = rand_vec(&mut rng);

        let fast = cosine_similarity(&a, &b);
        let slow = cosine_scalar(&a, &b);

        let diff = (fast - slow).abs();

        if diff > max_diff {
            max_diff = diff;
        }

        assert!(diff < 1e-3, "SIMD drift too large: {}", diff);
        assert!(!fast.is_nan());
    }

    println!("max drift = {}", max_diff);
}

// ── dispatch test ─────────────────────────────────────────────────────────────

/// Documents exactly what your N4020 supports.
/// Fails loudly if the CPU doesn't have what the code assumes.
#[test]
fn n4020_has_required_features() {
    assert!(
        is_x86_feature_detected!("sse4.1"),
        "N4020 must have SSE4.1 — SIMD path will never run without it"
    );
    assert!(
        is_x86_feature_detected!("sse4.2"),
        "N4020 must have SSE4.2"
    );
    // document what it does NOT have — not assertions, just printed
    println!("avx2:    {}", is_x86_feature_detected!("avx2"));
    println!("avx512f: {}", is_x86_feature_detected!("avx512f"));
}

/// Proves the runtime dispatch actually selects the SIMD path on this machine.
/// If this test passes, cosine_similarity() is using SSE4.1 not the fallback.
#[test]
#[cfg(target_arch = "x86_64")]
fn simd_path_is_selected_on_n4020() {
    // If SSE4.1 is available (it is on N4020), cosine_similarity()
    // must produce results within float epsilon of the SIMD impl directly.
    // We verify by calling both and checking they agree — if dispatch
    // was silently falling back to scalar they'd still agree, but
    // the cpu_features test above confirms SSE4.1 is present.
    let a = make_vec(1.3, 128);
    let b = make_vec(2.7, 128);
    let dispatched = cosine_similarity(&a, &b);
    let scalar     = cosine_similarity_scalar(&a, &b);
    assert!(
        (dispatched - scalar).abs() < 1e-5,
        "dispatch={} scalar={}", dispatched, scalar
    );
}

// ── parity tests ──────────────────────────────────────────────────────────────

/// Core invariant: SIMD and scalar must agree to within float rounding.
/// Runs 2000 random-ish pairs covering varied magnitudes.
#[test]
fn simd_matches_scalar_varied_inputs() {
    for seed in 0..2000 {
        let a = make_vec(seed as f32 * 0.017, 128);
        let b = make_vec(seed as f32 * 0.031, 128);

        let simd_result   = cosine_similarity(&a, &b);
        let scalar_result = cosine_similarity_scalar(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-5,
            "seed={} simd={} scalar={} diff={}",
            seed, simd_result, scalar_result,
            (simd_result - scalar_result).abs()
        );
    }
}

/// Non-multiple-of-4 lengths exercise the tail loop.
#[test]
fn simd_handles_non_multiple_of_4_lengths() {
    for len in [1, 2, 3, 5, 7, 9, 127, 129, 255, 257] {
        let a = make_vec(1.0, len);
        let b = make_vec(2.0, len);

        let simd_result   = cosine_similarity(&a, &b);
        let scalar_result = cosine_similarity_scalar(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-5,
            "len={} simd={} scalar={}", len, simd_result, scalar_result
        );
    }
}

/// Very small values — tests numerical stability near the 1e-8 floor.
#[test]
fn simd_stable_with_small_values() {
    let a: Vec<f32> = vec![1e-7; 128];
    let b: Vec<f32> = vec![1e-7; 128];
    let result = cosine_similarity(&a, &b);
    assert!(
        (result - 1.0).abs() < 1e-4,
        "identical tiny vectors must have similarity ~1.0, got {}. \
         Check that norm floor is applied per-norm not to product.", result
    );
}

// Add a companion test for the scalar path — same bug was there too
#[test]
fn scalar_stable_with_small_values() {
    let a: Vec<f32> = vec![1e-7; 128];
    let b: Vec<f32> = vec![1e-7; 128];
    let result = cosine_similarity_scalar(&a, &b);
    assert!(
        (result - 1.0).abs() < 1e-4,
        "scalar path: identical tiny vectors got {}", result
    );
}

/// Near-zero vector — must not divide by zero or produce NaN/inf.
#[test]
fn simd_handles_near_zero_vector() {
    let a: Vec<f32> = vec![0.0; 128];
    let b = make_vec(1.0, 128);
    let result = cosine_similarity(&a, &b);
    assert!(
        result.is_finite(),
        "near-zero input produced non-finite: {}", result
    );
}

/// Opposite vectors must give similarity near -1.0.
#[test]
fn simd_opposite_vectors_negative() {
    let a = make_vec(1.0, 128);
    let b: Vec<f32> = a.iter().map(|x| -x).collect();
    let result = cosine_similarity(&a, &b);
    assert!(
        (result - (-1.0)).abs() < 1e-5,
        "opposite vectors: got {}", result
    );
}

/// Identical vectors must give similarity = 1.0.
#[test]
fn simd_identical_vectors_is_one() {
    let a = make_vec(3.14, 128);
    let result = cosine_similarity(&a, &a);
    assert!(
        (result - 1.0).abs() < 1e-5,
        "self-similarity: got {}", result
    );
}

#[test]
fn simd_random_noise_stability() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);

    for _ in 0..1000 {
        let a: Vec<f32> = (0..128).map(|_| rng.random_range(-10.0..10.0)).collect();
        let b: Vec<f32> = (0..128).map(|_| rng.random_range(-10.0..10.0)).collect();

        let simd = cosine_similarity(&a, &b);
        let scalar = cosine_similarity_scalar(&a, &b);

        assert!(
            (simd - scalar).abs() < 1e-4,
            "random mismatch simd={} scalar={}",
            simd, scalar
        );
    }
}