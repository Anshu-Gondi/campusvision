use intelligence_core::embeddings::*;

fn w() { unsafe { std::env::set_var("FACE_DB_ROLE", "writer"); } }
fn emb(seed: f32) -> Vec<f32> {
    (0..128).map(|i| ((i as f32) * seed).sin()).collect()
}

#[test]
fn student_search_never_returns_teacher() {
    w();
    let school = "role_stu_not_teacher";
    let e = emb(1.0);
    // insert same embedding as both roles
    add_face_embedding(school, e.clone(), "Teacher".into(), 1, "T1".into(), "teacher".into()).unwrap();
    add_face_embedding(school, e.clone(), "Student".into(), 2, "S1".into(), "student".into()).unwrap();

    let results = search_in_role(school, &e, "student", 10);
    for (id, _) in &results {
        let meta = get_metadata(school, *id).unwrap();
        assert_eq!(meta.role, "student",
            "teacher appeared in student search: id={}", id);
    }
}

#[test]
fn teacher_search_never_returns_student() {
    w();
    let school = "role_tea_not_student";
    let e = emb(2.0);
    add_face_embedding(school, e.clone(), "Student".into(), 1, "S1".into(), "student".into()).unwrap();
    add_face_embedding(school, e.clone(), "Teacher".into(), 2, "T1".into(), "teacher".into()).unwrap();

    let results = search_in_role(school, &e, "teacher", 10);
    for (id, _) in &results {
        let meta = get_metadata(school, *id).unwrap();
        assert_eq!(meta.role, "teacher",
            "student appeared in teacher search: id={}", id);
    }
}

#[test]
fn count_by_role_is_accurate() {
    w();
    let school = "role_count_accurate";
    let s_before = count_by_role(school, "student");
    let t_before = count_by_role(school, "teacher");

    add_face_embedding(school, emb(1.0), "S1".into(), 1, "R1".into(), "student".into()).unwrap();
    add_face_embedding(school, emb(2.0), "S2".into(), 2, "R2".into(), "student".into()).unwrap();
    add_face_embedding(school, emb(3.0), "T1".into(), 3, "E1".into(), "teacher".into()).unwrap();

    assert_eq!(count_by_role(school, "student"), s_before + 2);
    assert_eq!(count_by_role(school, "teacher"), t_before + 1);
}

#[test]
fn role_case_insensitive() {
    w();
    let school = "role_case_insensitive";
    // "Teacher" and "TEACHER" and "teacher" all treated identically
    let id = add_face_embedding(school, emb(1.0), "T".into(),
        1, "E1".into(), "TEACHER".into()).unwrap();
    let meta = get_metadata(school, id).unwrap();
    // search as lowercase must find it
    let results = search_in_role(school, &emb(1.0), "teacher", 5);
    assert!(!results.is_empty(), "uppercase role not found by lowercase search");
}

#[test]
fn delete_affects_only_target_role_count() {
    w();
    let school = "role_delete_count";
    let sid = add_face_embedding(school, emb(1.0), "S".into(),
        1, "R1".into(), "student".into()).unwrap();
    add_face_embedding(school, emb(2.0), "T".into(),
        2, "E1".into(), "teacher".into()).unwrap();

    let t_before = count_by_role(school, "teacher");
    remove_face_embedding(school, sid).unwrap();

    // teacher count unchanged
    assert_eq!(count_by_role(school, "teacher"), t_before);
    // student count decreased
    assert_eq!(count_by_role(school, "student"), 0);
}