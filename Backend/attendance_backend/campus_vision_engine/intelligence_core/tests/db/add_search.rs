use std::env;

use intelligence_core::embeddings::*;

fn w() {
    unsafe {
        std::env::set_var("FACE_DB_ROLE", "writer");
    }
}

fn emb(seed: f32) -> Vec<f32> {
    (0..128).map(|i| ((i as f32) * seed).sin()).collect()
}

#[test]
fn add_returns_incrementing_ids() {
    w();
    let school = "db_add_ids";
    let id0 = add_face_embedding(
        school,
        emb(1.0),
        "A".into(),
        1,
        "R1".into(),
        "student".into()
    ).unwrap();
    let id1 = add_face_embedding(
        school,
        emb(2.0),
        "B".into(),
        2,
        "R2".into(),
        "student".into()
    ).unwrap();
    let id2 = add_face_embedding(
        school,
        emb(3.0),
        "C".into(),
        3,
        "R3".into(),
        "student".into()
    ).unwrap();
    assert_eq!(id1, id0 + 1);
    assert_eq!(id2, id1 + 1);
}

#[test]
fn metadata_survives_roundtrip() {
    w();
    let school = "db_meta_rt";
    let id = add_face_embedding(
        school,
        emb(1.1),
        "Rahul Sharma".into(),
        42,
        "R042".into(),
        "student".into()
    ).unwrap();
    let meta = get_metadata(school, id).unwrap();
    assert_eq!(meta.name, "Rahul Sharma");
    assert_eq!(meta.person_id, 42);
    assert_eq!(meta.roll_no, "R042");
    assert_eq!(meta.role, "student");
    assert!(!meta.deleted);
}

#[test]
fn search_finds_exact_embedding() {
    w();
    let school = "db_search_exact";
    let e = emb(2.2);
    add_face_embedding(school, e.clone(), "X".into(), 1, "R1".into(), "student".into()).unwrap();

    let results = search_in_role(school, &e, "student", 1);
    assert!(!results.is_empty(), "exact embedding not found");
    let (_, score) = results[0];
    assert!(score > 0.99, "self-search score too low: {}", score);
}

#[test]
fn search_respects_k_limit() {
    w();
    let school = "db_search_k";
    for i in 0..20 {
        add_face_embedding(
            school,
            emb((i as f32) * 0.1 + 0.1),
            "P".into(),
            i,
            format!("R{}", i),
            "student".into()
        ).unwrap();
    }
    let results = search_in_role(school, &emb(0.5), "student", 5);
    assert!(results.len() <= 5, "returned more than k={} results: {}", 5, results.len());
}

#[test]
fn scores_are_valid_range() {
    w();
    let school = "db_score_range";
    for i in 0..10 {
        add_face_embedding(
            school,
            emb((i as f32) * 0.3 + 0.1),
            "P".into(),
            i,
            format!("R{}", i),
            "student".into()
        ).unwrap();
    }
    let results = search_in_role(school, &emb(1.5), "student", 10);
    for (id, score) in &results {
        assert!(
            score.is_finite() && *score >= -1.01 && *score <= 1.01,
            "id={} score={} is out of valid range",
            id,
            score
        );
    }
}

#[test]
fn delete_removes_from_search() {
    w();
    let school = "db_delete_search";
    let e = emb(3.3);
    let id = add_face_embedding(
        school,
        e.clone(),
        "Del".into(),
        99,
        "R99".into(),
        "student".into()
    ).unwrap();

    remove_face_embedding(school, id).unwrap();

    let results = search_in_role(school, &e, "student", 10);
    assert!(!results.iter().any(|(rid, _)| *rid == id), "deleted face still appears in search");
}

#[test]
fn get_total_faces_counts_correctly() {
    w();
    let school = "db_total_count";
    let before = get_total_faces(school);
    add_face_embedding(school, emb(1.0), "A".into(), 1, "R1".into(), "student".into()).unwrap();
    add_face_embedding(school, emb(2.0), "B".into(), 2, "R2".into(), "teacher".into()).unwrap();
    assert_eq!(get_total_faces(school), before + 2);
}

#[test]
fn get_metadata_out_of_range_returns_none() {
    w();
    let school = "db_oob_meta";
    assert!(get_metadata(school, 99999).is_none());
}

#[test]
fn add_then_search_self_match() {
    unsafe {
        std::env::set_var("FACE_DB_ROLE", "writer");
    }

    clear_all(); // ✅

    let emb = vec![0.5; 128];

    let id = add_face_embedding(
        "school",
        emb.clone(),
        "A".into(),
        1,
        "1".into(),
        "student".into()
    ).unwrap();

    let res = search_in_role("school", &emb, "student", 1);

    assert!(!res.is_empty());

    let (found_id, score) = res[0];

    assert_eq!(found_id, id);
    assert!(score > 0.9);
}
