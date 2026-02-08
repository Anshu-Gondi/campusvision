use intelligence_core::embeddings;

#[derive(Debug, Clone)]
pub struct DuplicateCheckResult {
    pub duplicate: bool,
    pub matched_id: Option<usize>,
    pub similarity: Option<f32>,
    pub name: Option<String>,
    pub roll_no: Option<String>,
}

pub fn check_duplicate_rust(
    embedding: &[f32],
    role: &str,
    threshold: f32,
) -> DuplicateCheckResult {
    let results = embeddings::search_in_role(embedding, role, 1);

    if let Some((id, sim)) = results.first() {
        if *sim >= threshold {
            if let Some(meta) = embeddings::get_metadata(*id) {
                return DuplicateCheckResult {
                    duplicate: true,
                    matched_id: Some(*id),
                    similarity: Some(*sim),
                    name: Some(meta.name.clone()),
                    roll_no: Some(meta.roll_no.clone()),
                };
            }
        }
    }

    DuplicateCheckResult {
        duplicate: false,
        matched_id: None,
        similarity: None,
        name: None,
        roll_no: None,
    }
}

pub fn get_face_info_rust(id: usize) -> Option<(usize, String, String, String)> {
    embeddings::get_metadata(id).map(|meta| {
        (id, meta.name.clone(), meta.roll_no.clone(), meta.role.clone())
    })
}

pub fn count_students_rust() -> usize {
    embeddings::count_by_role("student")
}

pub fn count_teachers_rust() -> usize {
    embeddings::count_by_role("teacher")
}

pub fn save_database_rust(path: &str) -> anyhow::Result<()> {
    embeddings::save_all(path)?;
    Ok(())
}

pub fn load_database_rust(path: &str) -> anyhow::Result<()> {
    embeddings::load_all(path)?;
    Ok(())
}

pub fn total_registered_rust() -> usize {
    embeddings::get_total_faces()
}

pub fn init_database_rust() -> anyhow::Result<()> {
    embeddings::init_empty_database()?;
    Ok(())
}
