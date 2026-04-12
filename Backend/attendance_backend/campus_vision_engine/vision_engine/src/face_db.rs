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
    school_id: &str,
    embedding: &[f32],
    role: &str,
    threshold: f32,
) -> DuplicateCheckResult {

    let results = embeddings::search_in_role(
        school_id,
        embedding,
        role,
        1
    );

    if let Some((id, sim)) = results.first() {
        if *sim >= threshold {
            if let Some(meta) = embeddings::get_metadata(school_id, *id) {
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

pub fn get_face_info_rust(
    school_id: &str,
    id: usize
) -> Option<(usize, String, String, String)> {

    embeddings::get_metadata(school_id, id).map(|meta| {
        (id, meta.name.clone(), meta.roll_no.clone(), meta.role.clone())
    })
}

pub fn count_students_rust(school_id: &str) -> usize {
    embeddings::count_by_role(school_id, "student")
}

pub fn count_teachers_rust(school_id: &str) -> usize {
    embeddings::count_by_role(school_id, "teacher")
}

pub fn total_registered_rust(school_id: &str) -> usize {
    embeddings::get_total_faces(school_id)
}

pub fn save_database_rust(path: &str) -> anyhow::Result<()> {
    embeddings::save_all(path)?;
    Ok(())
}

pub fn load_database_rust(path: &str) -> anyhow::Result<()> {
    embeddings::load_all(path)?;
    Ok(())
}

pub fn init_database_rust(base_path: &str) -> anyhow::Result<()> {
    // 🔥 1. CLEAR MEMORY
    intelligence_core::embeddings::clear_all();

    // 🔥 2. CREATE EMPTY SNAPSHOT STRUCTURE
    std::fs::create_dir_all(base_path)?;

    // 🔥 3. FORCE SAVE (empty DB)
    intelligence_core::embeddings::force_save_all(base_path)?;

    Ok(())
}