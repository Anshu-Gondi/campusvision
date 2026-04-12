use intelligence_core::embeddings;

/// Add a person embedding into the index
pub fn add_person_rust(
    school_id: &str, // 🔥 ADD THIS
    embedding: Vec<f32>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String
) -> Result<usize, String> {
    if !["student", "teacher"].contains(&role.as_str()) {
        return Err("role must be 'student' or 'teacher'".into());
    }

    embeddings
        ::add_face_embedding(
            school_id, // 🔥 CRITICAL
            embedding,
            name,
            person_id,
            roll_no,
            role
        )
        .map_err(|e| e.to_string())
}

/// Search for similar people
pub fn search_person_rust(
    school_id: &str, // 🔥 ADD
    embedding: Vec<f32>,
    role: String,
    k: usize
) -> Result<Vec<(usize, f32)>, String> {
    if !["student", "teacher"].contains(&role.as_str()) {
        return Err("role must be 'student' or 'teacher'".into());
    }

    Ok(
        embeddings::search_in_role(
            school_id, // 🔥 CRITICAL
            &embedding,
            &role,
            k
        )
    )
}

/// Add embedding directly to index
pub fn add_to_index_rust(
    school_id: &str, // 🔥 ADD
    embedding: Vec<f32>,
    person_id: u64,
    name: String,
    roll_no: String,
    role: String
) -> Result<usize, String> {
    embeddings
        ::add_face_embedding(
            school_id, // 🔥 CRITICAL
            embedding,
            name,
            person_id,
            roll_no,
            role
        )
        .map_err(|e| e.to_string())
}

/// Query similar students (IDs only)
pub fn query_similar_rust(
    school_id: &str, // 🔥 ADD
    embedding: Vec<f32>,
    k: usize
) -> Vec<usize> {
    embeddings
        ::search_in_role(
            school_id, // 🔥 CRITICAL
            &embedding,
            "student",
            k
        )
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}
