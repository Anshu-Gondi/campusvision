use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::RngExt;

use intelligence_core::utils::normalize;

#[test]
fn normalization_invariants() {
    let mut rng = StdRng::seed_from_u64(7);

    for _ in 0..20_000 {
        let mut v: Vec<f32> = (0..128).map(|_| rng.random_range(-10.0..10.0)).collect();

        normalize(&mut v);

        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();

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