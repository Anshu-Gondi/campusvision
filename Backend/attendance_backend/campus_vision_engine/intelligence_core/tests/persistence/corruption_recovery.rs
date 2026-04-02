use std::env;
use std::fs::OpenOptions;
use std::io::Write;

use rand::RngExt;
use rand::{ Rng, SeedableRng };
use rand::rngs::StdRng;
use tempfile::tempdir;

use intelligence_core::embeddings::*;

#[test]
fn corrupted_file_should_not_kill_system() {
    unsafe {
        std::env::set_var("FACE_DB_ROLE", "writer");
    }

    let dir = tempdir().unwrap();
    let base = dir.path().to_str().unwrap();

    let mut rng = StdRng::seed_from_u64(1);

    for _ in 0..3000 {
        add_face_embedding(
            "school",
            (0..128).map(|_| rng.random()).collect(),
            "A".into(),
            rng.random(),
            "1".into(),
            "student".into()
        ).unwrap();
    }

    save_all(base).unwrap();

    // corrupt file
    let file_path = dir.path().join("data.bin");
    let mut f = OpenOptions::new().write(true).open(&file_path).unwrap();
    f.write_all(&[1, 2, 3, 4, 5]).unwrap();

    // must not panic
    let _ = load_all(base);

    let res = search_in_role("school", &(0..128).map(|_| 0.5).collect::<Vec<_>>(), "student", 5);

    assert!(res.len() <= 5);
}
