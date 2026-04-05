use std::thread;
use std::env;

use rand::{ Rng, SeedableRng };
use rand::rngs::StdRng;
use rand::RngExt;


use intelligence_core::embeddings::*;
use std::sync::{Arc, Barrier};


const THREADS: usize = 4;

fn w() { unsafe { std::env::set_var("FACE_DB_ROLE", "writer"); } }

fn rand_emb(rng: &mut StdRng) -> Vec<f32> {
    (0..128).map(|_| rng.random()).collect()
}


#[test]
fn readers_never_see_corrupt_scores() {
    w();
    let school = "rw_race_scores";

    // pre-populate so readers have something to find
    let mut rng = StdRng::seed_from_u64(0);
    for i in 0..50 {
        add_face_embedding(school, rand_emb(&mut rng), "P".into(),
            i, format!("R{}", i), "student".into()).unwrap();
    }

    let n_writers = 4;
    let n_readers = 4;
    let barrier = Arc::new(Barrier::new(n_writers + n_readers));

    let mut handles = vec![];

    for t in 0..n_writers {
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64 * 100);
            b.wait();
            for i in 0..300 {
                let _ = add_face_embedding(
                    school, rand_emb(&mut rng), "W".into(),
                    (t * 1000 + i) as u64, "R".into(), "student".into()
                );
            }
        }));
    }

    for t in 0..n_readers {
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64 * 200 + 1);
            b.wait();
            for _ in 0..300 {
                let q = rand_emb(&mut rng);
                let results = search_in_role(school, &q, "student", 5);
                for (id, score) in results {
                    assert!(
                        score.is_finite() && score >= -1.01 && score <= 1.01,
                        "corrupt score {} for id {}", score, id
                    );
                }
            }
        }));
    }

    for h in handles { h.join().expect("thread panicked"); }
}

#[test]
fn batch_search_concurrent_with_inserts() {
    w();
    let school = "rw_batch_search";
    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..30 {
        add_face_embedding(school, rand_emb(&mut rng), "P".into(),
            i, format!("R{}", i), "student".into()).unwrap();
    }

    let barrier = Arc::new(Barrier::new(3));
    let mut handles = vec![];

    // writer
    {
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(1);
            b.wait();
            for i in 0..200 {
                let _ = add_face_embedding(school, rand_emb(&mut rng),
                    "W".into(), (5000 + i) as u64, "R".into(), "student".into());
            }
        }));
    }

    // two batch searchers
    for t in 0..2 {
        let b = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64 * 77 + 2);
            b.wait();
            for _ in 0..100 {
                let queries: Vec<Vec<f32>> = (0..5).map(|_| rand_emb(&mut rng)).collect();
                let results = batch_search(school, &queries, "student", 3);
                assert_eq!(results.len(), 5);
                for r in results {
                    if let Some((_, score)) = r {
                        assert!(score.is_finite(),
                            "batch_search returned NaN/inf score");
                    }
                }
            }
        }));
    }

    for h in handles { h.join().expect("thread panicked"); }
}



#[test]
fn heavy_read_write_race() {
    w();

    clear_all(); // 🔥 REQUIRED

    let school = "race_school";
    let mut handles = vec![];

    for t in 0..THREADS {
        let school = school.to_string();

        let h = thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64);

            for _ in 0..2000 {
                let op: u8 = rng.random_range(0..4);

                match op {
                    // 🔹 WRITE
                    0 => {
                        let _ = add_face_embedding(
                            &school,
                            rand_emb(&mut rng),
                            "X".into(),
                            rng.random(),
                            "r".into(),
                            "student".into()
                        );
                    }

                    // 🔹 SEARCH
                    1 => {
                        let q = rand_emb(&mut rng);
                        let res = search_in_role(&school, &q, "student", 5);

                        for (id, _) in res {
                            // ✅ validate id exists
                            if let Some(meta) = get_metadata(&school, id) {
                                assert!(!meta.deleted);
                                assert_eq!(meta.role.to_lowercase(), "student");
                            } else {
                                panic!("Returned ID does not exist");
                            }
                        }
                    }

                    // 🔹 COUNT
                    2 => {
                        let total = get_total_faces(&school);

                        // ✅ sanity check
                        assert!(total >= 0);
                    }

                    // 🔹 DELETE (NEW — CRITICAL)
                    _ => {
                        let total = get_total_faces(&school);

                        if total > 0 {
                            let id = rng.random_range(0..total);
                            let _ = remove_face_embedding(&school, id);
                        }
                    }
                }
            }
        });

        handles.push(h);
    }

    for h in handles {
        h.join().unwrap();
    }
}