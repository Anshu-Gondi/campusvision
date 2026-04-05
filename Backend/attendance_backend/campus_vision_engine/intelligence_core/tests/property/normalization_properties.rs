use rand::{ Rng, SeedableRng };
use rand::rngs::StdRng;
use rand::RngExt;

use intelligence_core::utils::{
    normalize,
    make_vec,
    scalar_norm,
    cosine_similarity,
    cosine_similarity_scalar,
};

#[test]
fn normalization_invariants() {
    let mut rng = StdRng::seed_from_u64(7);

    for _ in 0..20_000 {
        let mut v: Vec<f32> = (0..128).map(|_| rng.random_range(-10.0..10.0)).collect();

        normalize(&mut v);

        let norm = v
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        assert!((norm - 1.0).abs() < 1e-3 || norm == 0.0);
        assert!(v.iter().all(|x: &f32| !x.is_nan()));
    }
}

#[test]
fn normalization_zero_vector_stability() {
    let mut v = vec![0.0; 128];

    normalize(&mut v);

    // invariant: should not produce NaN
    assert!(v.iter().all(|x: &f32| !x.is_nan()));
}

// ── normalize tests ───────────────────────────────────────────────────────────

/// After normalize(), L2 norm must be 1.0 within float epsilon.
#[test]
fn normalize_produces_unit_vector() {
    for seed in 0..500 {
        let mut v = make_vec((seed as f32) * 0.13 + 0.1, 128);
        normalize(&mut v);
        let norm = scalar_norm(&v);
        assert!((norm - 1.0).abs() < 1e-5, "seed={} norm={} after normalize", seed, norm);
    }
}

/// normalize() then cosine_similarity(v, v) must be exactly 1.0.
#[test]
fn normalized_self_similarity_is_one() {
    let mut v = make_vec(42.0, 128);
    normalize(&mut v);
    let sim = cosine_similarity(&v, &v);
    assert!((sim - 1.0).abs() < 1e-5, "normalized self-similarity: {}", sim);
}

/// normalize() is idempotent — calling it twice changes nothing.
#[test]
fn normalize_is_idempotent() {
    let mut v = make_vec(7.77, 128);
    normalize(&mut v);
    let once = v.clone();
    normalize(&mut v);
    for (a, b) in once.iter().zip(v.iter()) {
        assert!((*a - *b).abs() < 1e-6, "normalize not idempotent: {} vs {}", a, b);
    }
}
