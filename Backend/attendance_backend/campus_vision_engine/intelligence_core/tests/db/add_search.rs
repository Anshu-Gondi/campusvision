use std::env;

use intelligence_core::embeddings::*;

#[test]
fn add_then_search_self_match() {
    unsafe {
        std::env::set_var("FACE_DB_ROLE", "writer");
    }

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

    // invariant: self match should be highest similarity
    assert_eq!(found_id, id);
    assert!(score > 0.9);
}
