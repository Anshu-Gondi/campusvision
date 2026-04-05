use std::env;
use std::fs::OpenOptions;
use std::io::Write;

use rand::RngExt;
use rand::{ Rng, SeedableRng };
use rand::rngs::StdRng;
use tempfile::tempdir;

use intelligence_core::embeddings::*;
use std::fs;

fn w() { unsafe { std::env::set_var("FACE_DB_ROLE", "writer"); } }

#[test]
fn corrupted_file_returns_ok_not_panic() {
    w();
    let path = "./test_persist_corrupt";
    let _ = fs::remove_dir_all(path);
    fs::create_dir_all(path).unwrap();
    fs::write(format!("{}/data.bin", path), b"not valid bincode").unwrap();

    // your load_all silently returns Ok(()) on deserialize error
    // this test confirms it doesn't panic
    let result = load_all(path);
    assert!(result.is_ok(), "corrupted file should not panic: {:?}", result);

    let _ = fs::remove_dir_all(path);
}

#[test]
fn missing_file_returns_ok() {
    w();
    // non-existent path — must return Ok silently
    let result = load_all("./definitely_does_not_exist_12345");
    assert!(result.is_ok());
}

#[test]
fn empty_dir_returns_ok() {
    w();
    let path = "./test_persist_empty_dir";
    let _ = fs::remove_dir_all(path);
    fs::create_dir_all(path).unwrap();
    // directory exists but no data.bin
    let result = load_all(path);
    assert!(result.is_ok());
    let _ = fs::remove_dir_all(path);
}

#[test]
fn corrupted_file_should_not_kill_system() {
    w();

    clear_all(); // ✅

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

    // 🔥 simulate restart
    clear_all();

    // corrupt file
    let file_path = dir.path().join("data.bin");
    let mut f = OpenOptions::new().write(true).open(&file_path).unwrap();
    f.write_all(&[1, 2, 3, 4, 5]).unwrap();

    // must not panic
    let _ = load_all(base);

    // system should still be usable
    let res = search_in_role(
        "school",
        &(0..128).map(|_| 0.5).collect::<Vec<_>>(),
        "student",
        5
    );

    assert!(res.len() <= 5);
}