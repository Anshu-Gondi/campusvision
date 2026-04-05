use intelligence_core::embeddings::*;
use std::sync::{Arc, Barrier};
use std::thread;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::RngExt;

fn w() { unsafe { std::env::set_var("FACE_DB_ROLE", "writer"); } }
fn emb(rng: &mut StdRng) -> Vec<f32> {
    (0..128).map(|_| rng.random()).collect()
}

#[test]
fn multiple_schools_concurrent_no_cross_contamination() {
    w();
    let schools: Vec<&str> = vec![
        "shard_c_school_0", "shard_c_school_1",
        "shard_c_school_2", "shard_c_school_3",
    ];
    let n = schools.len();
    let barrier = Arc::new(Barrier::new(n));

    let handles: Vec<_> = (0..n).map(|t| {
        let b = Arc::clone(&barrier);
        let school = schools[t];
        thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64 * 31);
            b.wait();
            for i in 0..150 {
                add_face_embedding(school, emb(&mut rng),
                    "P".into(), i as u64, format!("R{}", i),
                    "student".into()).unwrap();
            }
        })
    }).collect();

    for h in handles { h.join().expect("thread panicked"); }

    // each school has exactly 150 faces — no cross-contamination
    for school in &schools {
        assert_eq!(get_total_faces(school), 150,
            "school {} has wrong count", school);
    }
}