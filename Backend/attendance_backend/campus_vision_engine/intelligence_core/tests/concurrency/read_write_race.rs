use std::sync::Arc;
use std::thread;
use std::env;

use rand::{ Rng, SeedableRng };
use rand::rngs::StdRng;
use rand::RngExt;


use intelligence_core::embeddings::*;

const THREADS: usize = 10;

fn rand_emb(rng: &mut StdRng) -> Vec<f32> {
    (0..128).map(|_| rng.random()).collect()
}

#[test]
fn heavy_read_write_race() {
    unsafe {
        std::env::set_var("FACE_DB_ROLE", "writer");
    }

    let school = "race_school";
    let mut handles = vec![];

    for t in 0..THREADS {
        let school = school.to_string();

        let h = thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(t as u64);

            for _ in 0..5000 {
                let op: u8 = rng.random_range(0..3);

                match op {
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
                    1 => {
                        let q = rand_emb(&mut rng);
                        let res = search_in_role(&school, &q, "student", 5);

                        assert!(res.len() <= 5);
                    }
                    _ => {
                        let total = get_total_faces(&school);
                        assert!(total >= 0);
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
