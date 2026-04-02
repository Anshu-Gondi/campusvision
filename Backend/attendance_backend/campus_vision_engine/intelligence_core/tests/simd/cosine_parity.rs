use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::RngExt;

use intelligence_core::utils::cosine_similarity;

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