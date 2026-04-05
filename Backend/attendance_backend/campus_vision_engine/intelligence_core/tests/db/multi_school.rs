use std::env;
use rand::{ Rng, SeedableRng };
use rand::rngs::StdRng;
use rand::RngExt;

use intelligence_core::embeddings::*;

fn rand_emb(rng: &mut StdRng) -> Vec<f32> {
    (0..128).map(|_| rng.random()).collect()
}

fn w() { unsafe { std::env::set_var("FACE_DB_ROLE", "writer"); } }

fn emb(seed: f32) -> Vec<f32> {
    (0..128).map(|i| ((i as f32) * seed).sin()).collect()
}

#[test]
fn schools_are_isolated() {
    w();
    let school_a = "multi_school_a_iso";
    let school_b = "multi_school_b_iso";

    let e = emb(1.0);
    add_face_embedding(school_a, e.clone(), "In A".into(), 1, "R1".into(), "student".into()).unwrap();

    // school_b should not find school_a's face
    let results = search_in_role(school_b, &e, "student", 5);
    // either empty or contains only school_b's own data
    assert_eq!(get_total_faces(school_b), 0,
        "school_b has faces that belong to school_a");
}

#[test]
fn same_person_id_in_different_schools_no_conflict() {
    w();
    let school_a = "multi_pid_a";
    let school_b = "multi_pid_b";

    // same person_id=1 in both schools — must not conflict
    add_face_embedding(school_a, emb(1.0), "Alice in A".into(),
        1, "R1".into(), "student".into()).unwrap();
    add_face_embedding(school_b, emb(2.0), "Alice in B".into(),
        1, "R1".into(), "student".into()).unwrap();

    // each school sees only its own data
    assert_eq!(get_total_faces(school_a), 1);
    assert_eq!(get_total_faces(school_b), 1);

    let meta_a = get_embeddings_for_person(school_a, 1, "student").unwrap();
    let meta_b = get_embeddings_for_person(school_b, 1, "student").unwrap();
    assert_eq!(meta_a.len(), 1);
    assert_eq!(meta_b.len(), 1);
}

#[test]
fn delete_in_one_school_does_not_affect_other() {
    w();
    let school_a = "multi_del_a";
    let school_b = "multi_del_b";

    let id_a = add_face_embedding(school_a, emb(1.0), "A".into(),
        1, "R1".into(), "student".into()).unwrap();
    add_face_embedding(school_b, emb(1.0), "B".into(),
        1, "R1".into(), "student".into()).unwrap();

    remove_face_embedding(school_a, id_a).unwrap();

    assert_eq!(get_total_faces(school_a), 0);
    assert_eq!(get_total_faces(school_b), 1, "delete in school_a affected school_b");
}

#[test]
fn many_schools_simultaneously() {
    w();
    let schools: Vec<String> = (0..10)
        .map(|i| format!("multi_many_school_{}", i))
        .collect();

    for (i, school) in schools.iter().enumerate() {
        add_face_embedding(school, emb(i as f32 + 0.1), "P".into(),
            i as u64, "R1".into(), "student".into()).unwrap();
    }

    for (i, school) in schools.iter().enumerate() {
        assert_eq!(get_total_faces(school), 1,
            "school {} has wrong face count", i);
    }
}

#[test]
fn strict_isolation_between_schools() {
    unsafe {
        std::env::set_var("FACE_DB_ROLE", "writer");
    }

    clear_all();

    let mut rng = StdRng::seed_from_u64(99);

    // Track person_ids per school to verify isolation properly
    let mut school_a_ids = std::collections::HashSet::new();
    let mut school_b_ids = std::collections::HashSet::new();

    for _ in 0..2000 {
        let person_a = rng.random();
        let person_b = rng.random();

        school_a_ids.insert(person_a);
        school_b_ids.insert(person_b);

        add_face_embedding(
            "school_a",
            rand_emb(&mut rng),
            "A".into(),
            person_a,
            "1".into(),
            "student".into(),
        ).unwrap();

        add_face_embedding(
            "school_b",
            rand_emb(&mut rng),
            "B".into(),
            person_b,
            "2".into(),
            "teacher".into(),
        ).unwrap();
    }

    let query = rand_emb(&mut rng);

    let res_a = search_in_role("school_a", &query, "student", 10);
    let res_b = search_in_role("school_b", &query, "teacher", 10);

    // ✅ Validate school A results
    for (id, _) in res_a {
        let meta = get_metadata("school_a", id).unwrap();

        // role check
        assert_eq!(meta.role.to_lowercase(), "student");

        // ✅ belongs to school A dataset
        assert!(school_a_ids.contains(&meta.person_id));

        // ❌ MUST NOT belong to school B dataset
        assert!(!school_b_ids.contains(&meta.person_id));
    }

    // ✅ Validate school B results
    for (id, _) in res_b {
        let meta = get_metadata("school_b", id).unwrap();

        // role check
        assert_eq!(meta.role.to_lowercase(), "teacher");

        // ✅ belongs to school B dataset
        assert!(school_b_ids.contains(&meta.person_id));

        // ❌ MUST NOT belong to school A dataset
        assert!(!school_a_ids.contains(&meta.person_id));
    }
}