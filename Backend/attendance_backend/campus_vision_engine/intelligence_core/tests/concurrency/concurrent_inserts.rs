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
fn concurrent_inserts_no_panic_or_corruption() {
    w();
    let school = "conc_insert_no_panic";
    let n_threads = 8;
    let barrier = Arc::new(Barrier::new(n_threads));

    let handles: Vec<_> = (0..n_threads).map(|t| {
        let b = Arc::clone(&barrier);
        thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64 * 1337);
            b.wait(); // all fire simultaneously
            for i in 0..200 {
                let role = if i % 2 == 0 { "student" } else { "teacher" };
                add_face_embedding(
                    school, emb(&mut rng),
                    format!("T{}P{}", t, i), (t * 1000 + i) as u64,
                    format!("R{}{}", t, i), role.into()
                ).unwrap();
            }
        })
    }).collect();

    for h in handles { h.join().expect("thread panicked"); }

    // total must be exactly what was inserted
    let total = get_total_faces(school);
    assert_eq!(total, n_threads * 200,
        "expected {} faces, got {}", n_threads * 200, total);
}

#[test]
fn mixed_role_inserts_count_correctly() {
    w();
    let school = "conc_mixed_roles";
    let barrier = Arc::new(Barrier::new(4));

    let handles: Vec<_> = (0..4).map(|t| {
        let b = Arc::clone(&barrier);
        thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64 * 999);
            b.wait();
            for i in 0..100 {
                // threads 0,1 insert students; threads 2,3 insert teachers
                let role = if t < 2 { "student" } else { "teacher" };
                add_face_embedding(
                    school, emb(&mut rng),
                    "X".into(), (t * 500 + i) as u64,
                    format!("R{}{}", t, i), role.into()
                ).unwrap();
            }
        })
    }).collect();

    for h in handles { h.join().expect("thread panicked"); }

    assert_eq!(count_by_role(school, "student"), 200);
    assert_eq!(count_by_role(school, "teacher"), 200);
}