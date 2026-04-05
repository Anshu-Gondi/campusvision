use intelligence_core::embeddings::*;
use std::fs;

const PATH: &str = "./test_persist_save_load";

fn w() { unsafe { std::env::set_var("FACE_DB_ROLE", "writer"); } }
fn emb(seed: f32) -> Vec<f32> {
    (0..128).map(|i| ((i as f32) * seed).sin()).collect()
}

fn cleanup() { let _ = fs::remove_dir_all(PATH); }

#[test]
fn full_save_reload_preserves_state() {
    w();
    cleanup();
    let school = "persist_full_save";

    let ids: Vec<usize> = (0..10).map(|i| {
        add_face_embedding(school, emb(i as f32 * 0.3 + 0.1),
            format!("Person_{}", i), i as u64,
            format!("R{:03}", i), "student".into()).unwrap()
    }).collect();

    // bypass 10s interval
    force_save_all(PATH).unwrap();
    clear_all();
    load_all(PATH).unwrap();

    for (i, &id) in ids.iter().enumerate() {
        let meta = get_metadata(school, id)
            .unwrap_or_else(|| panic!("id {} missing after reload", id));
        assert_eq!(meta.person_id, i as u64);
        assert_eq!(meta.name, format!("Person_{}", i));
        assert!(!meta.deleted);
    }

    // search still works after reload
    let results = search_in_role(school, &emb(0.1), "student", 3);
    assert!(!results.is_empty(), "search returned nothing after reload");

    cleanup();
}

#[test]
fn deleted_faces_not_reloaded() {
    w();
    cleanup();
    let school = "persist_delete_not_reload";

    let keep = add_face_embedding(school, emb(1.0), "Keep".into(),
        1, "R1".into(), "student".into()).unwrap();
    let del  = add_face_embedding(school, emb(2.0), "Delete".into(),
        2, "R2".into(), "student".into()).unwrap();

    remove_face_embedding(school, del).unwrap();
    force_save_all(PATH).unwrap();
    clear_all();
    load_all(PATH).unwrap();

    // kept face survives
    let meta = get_metadata(school, keep).unwrap();
    assert!(!meta.deleted, "kept face marked deleted after reload");

    // deleted face does not appear in search
    let results = search_in_role(school, &emb(2.0), "student", 10);
    assert!(!results.iter().any(|(id, _)| *id == del),
        "deleted face appeared in search after reload");

    cleanup();
}

#[test]
fn teacher_and_student_both_survive_reload() {
    w();
    cleanup();
    let school = "persist_both_roles";

    let sid = add_face_embedding(school, emb(1.0), "Student".into(),
        1, "R1".into(), "student".into()).unwrap();
    let tid = add_face_embedding(school, emb(2.0), "Teacher".into(),
        2, "E1".into(), "teacher".into()).unwrap();

    force_save_all(PATH).unwrap();
    clear_all();
    load_all(PATH).unwrap();

    assert_eq!(count_by_role(school, "student"), 1);
    assert_eq!(count_by_role(school, "teacher"), 1);

    let smeta = get_metadata(school, sid).unwrap();
    let tmeta = get_metadata(school, tid).unwrap();
    assert_eq!(smeta.role, "student");
    assert_eq!(tmeta.role, "teacher");

    cleanup();
}