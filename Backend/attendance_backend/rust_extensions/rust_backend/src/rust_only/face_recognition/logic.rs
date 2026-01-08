use crate::hnsw_helper;

/// Add a person embedding into the index
pub fn add_person_rust(
    embedding: Vec<f32>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String,
) -> Result<usize, String> {
    if !["student", "teacher"].contains(&role.as_str()) {
        return Err("role must be 'student' or 'teacher'".into());
    }

    hnsw_helper::add_face_embedding(
        embedding,
        name,
        person_id,
        roll_no,
        role,
    )
    .map_err(|e| e.to_string())
}

/// Search for similar people
pub fn search_person_rust(
    embedding: Vec<f32>,
    role: String,
    k: usize,
) -> Result<Vec<(usize, f32)>, String> {
    if !["student", "teacher"].contains(&role.as_str()) {
        return Err("role must be 'student' or 'teacher'".into());
    }

    Ok(hnsw_helper::search_in_role(&embedding, &role, k))
}

/// Check if a person can be re-enrolled
pub fn can_reenroll_rust(
    embedding: Vec<f32>,
    person_id: u64,
    role: String,
) -> Result<bool, String> {
    hnsw_helper::can_reenroll(&embedding, person_id, &role)
        .map_err(|e| e.to_string())
}

/// Add embedding directly to index
pub fn add_to_index_rust(
    embedding: Vec<f32>,
    person_id: u64,
    name: String,
    roll_no: String,
    role: String,
) -> Result<usize, String> {
    hnsw_helper::add_face_embedding(
        embedding,
        name,
        person_id,
        roll_no,
        role,
    )
    .map_err(|e| e.to_string())
}

/// Query similar students (IDs only)
pub fn query_similar_rust(
    embedding: Vec<f32>,
    k: usize,
) -> Vec<usize> {
    hnsw_helper::search_in_role(&embedding, "student", k)
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}
