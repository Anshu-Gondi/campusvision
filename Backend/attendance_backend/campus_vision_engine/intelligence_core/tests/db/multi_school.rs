use std::env;
use rand::{ Rng, SeedableRng };
use rand::rngs::StdRng;
use rand::RngExt;

use intelligence_core::embeddings::*;

fn rand_emb(rng: &mut StdRng) -> Vec<f32> {
    (0..128).map(|_| rng.random()).collect()
}

#[test]
fn strict_isolation_between_schools() {
    unsafe {
        std::env::set_var("FACE_DB_ROLE", "writer");
    }

    let mut rng = StdRng::seed_from_u64(99);

    for _ in 0..2000 {
        add_face_embedding(
            "school_a",
            rand_emb(&mut rng),
            "A".into(),
            rng.random(),
            "1".into(),
            "student".into()
        ).unwrap();

        add_face_embedding(
            "school_b",
            rand_emb(&mut rng),
            "B".into(),
            rng.random(),
            "2".into(),
            "teacher".into()
        ).unwrap();
    }

    let query = rand_emb(&mut rng);

    let res_a = search_in_role("school_a", &query, "student", 10);
    let res_b = search_in_role("school_b", &query, "teacher", 10);

    for (id, _) in res_a {
        let meta = get_metadata("school_a", id).unwrap();
        assert_eq!(meta.role.to_lowercase(), "student");
    }

    for (id, _) in res_b {
        let meta = get_metadata("school_b", id).unwrap();
        assert_eq!(meta.role.to_lowercase(), "teacher");
    }
}
