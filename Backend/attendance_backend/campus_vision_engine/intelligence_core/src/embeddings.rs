use crate::metadata::FaceMetadata;
use crate::utils::cosine_similarity;
use anyhow::Result;
use hnsw_rs::prelude::*;
use lru::LruCache;
use once_cell::sync::Lazy;
use serde::{ Deserialize, Serialize };
use std::collections::HashMap;
use std::fs::{ self, File };
use std::io::{ BufWriter, Write };
use std::path::Path;
use std::sync::{ Mutex, OnceLock, RwLock };
use std::time::{ Duration, Instant };
use std::num::NonZeroUsize;

/// Writer guard
fn is_writer() -> bool {
    std::env
        ::var("FACE_DB_ROLE")
        .map(|v| v == "writer")
        .unwrap_or(false)
}

// HNSW <'static, f32, DistCosine> - fully Send + Sync for Python FFI
type IndexType = Hnsw<'static, f32, DistCosine>;

static NEXT_ID: Lazy<Mutex<usize>> = Lazy::new(|| Mutex::new(0));

/// Per-school indices (student & teacher)
static SCHOOL_INDICES: Lazy<RwLock<HashMap<String, (IndexType, IndexType)>>> = Lazy::new(||
    RwLock::new(HashMap::new())
);

/// Metadata and embeddings globally, keyed by school
static SCHOOL_METADATA: Lazy<RwLock<HashMap<String, HashMap<usize, FaceMetadata>>>> = Lazy::new(||
    RwLock::new(HashMap::new())
);

static SCHOOL_EMBEDDINGS: Lazy<RwLock<HashMap<String, HashMap<usize, Vec<f32>>>>> = Lazy::new(||
    RwLock::new(HashMap::new())
);

/// Short-term frame-to-frame cache per school (bounded)
static LAST_EMBEDDINGS: Lazy<RwLock<HashMap<String, LruCache<String, Vec<f32>>>>> = Lazy::new(||
    RwLock::new(HashMap::new())
);

static LAST_SAVE: OnceLock<Mutex<Instant>> = OnceLock::new();
const SAVE_INTERVAL: Duration = Duration::from_secs(10);

fn should_save() -> bool {
    let m = LAST_SAVE.get_or_init(|| Mutex::new(Instant::now() - SAVE_INTERVAL));
    let mut last = m.lock().unwrap();

    if last.elapsed() >= SAVE_INTERVAL {
        *last = Instant::now();
        true
    } else {
        false
    }
}

/// Initialize per-school indices
fn init_school_indices(school_id: &str) {
    let mut idx_map = SCHOOL_INDICES.write().unwrap();
    idx_map.entry(school_id.to_string()).or_insert_with(|| {
        (
            Hnsw::new(16, 50_000, 100, 50, DistCosine), // student
            Hnsw::new(16, 5_000, 100, 50, DistCosine), // teacher
        )
    });

    let mut meta_map = SCHOOL_METADATA.write().unwrap();
    meta_map.entry(school_id.to_string()).or_insert_with(HashMap::new);

    let mut em_map = SCHOOL_EMBEDDINGS.write().unwrap();
    em_map.entry(school_id.to_string()).or_insert_with(HashMap::new);

    let mut cache_map = LAST_EMBEDDINGS.write().unwrap();
    cache_map
        .entry(school_id.to_string())
        .or_insert_with(|| { LruCache::new(NonZeroUsize::new(1000).unwrap()) }); // limit per school
}

/// Add embedding for a school
pub fn add_face_embedding(
    school_id: &str,
    embedding: Vec<f32>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String
) -> Result<usize> {
    if !is_writer() {
        anyhow::bail!("This process is not allowed to modify face database");
    }

    init_school_indices(school_id);

    let role_lower = role.to_lowercase();
    let is_teacher = role_lower == "teacher";

    // Generate ID
    let mut counter = NEXT_ID.lock().unwrap();
    let id = *counter;
    *counter += 1;

    // Insert into proper index
    {
        let mut idx_map = SCHOOL_INDICES.write().unwrap();
        let (s_idx, t_idx) = idx_map.get_mut(school_id).unwrap();
        if is_teacher {
            t_idx.insert((&embedding[..], id));
        } else {
            s_idx.insert((&embedding[..], id));
        }
    }

    // Store metadata
    {
        let mut meta_map = SCHOOL_METADATA.write().unwrap();
        let meta = meta_map.get_mut(school_id).unwrap();
        meta.insert(id, FaceMetadata {
            id: id.to_string(),
            name,
            person_id,
            roll_no,
            role,
            reliability: None,
        });
    }

    // Store embedding
    {
        let mut em_map = SCHOOL_EMBEDDINGS.write().unwrap();
        let em = em_map.get_mut(school_id).unwrap();
        em.insert(id, embedding.clone());
    }

    // Add to LRU cache
    {
        let mut cache_map = LAST_EMBEDDINGS.write().unwrap();
        let cache = cache_map.get_mut(school_id).unwrap();
        cache.put(format!("{}:{}", role_lower, person_id), embedding);
    }

    Ok(id)
}

/// Search within a school & role
pub fn search_in_role(
    school_id: &str,
    embedding: &[f32],
    role: &str,
    k: usize
) -> Vec<(usize, f32)> {
    init_school_indices(school_id);
    let ef_search = 100;
    let idx_map = SCHOOL_INDICES.read().unwrap();
    let (s_idx, t_idx) = idx_map.get(school_id).unwrap();
    let idx = if role.to_lowercase() == "teacher" { t_idx } else { s_idx };

    idx.search(embedding, k, ef_search)
        .into_iter()
        .map(|n| (n.d_id, 1.0 - n.distance))
        .collect()
}

/// Save all schools to disk
#[derive(Serialize, Deserialize)]
struct SerializableData {
    version: u8,
    school_data: HashMap<String, SchoolSerializable>,
}

#[derive(Serialize, Deserialize)]
struct SchoolSerializable {
    student_embeddings: Vec<(Vec<f32>, usize)>,
    teacher_embeddings: Vec<(Vec<f32>, usize)>,
    metadata: HashMap<usize, FaceMetadata>,
}

pub fn save_all(base_path: &str) -> Result<()> {
    if !is_writer() {
        anyhow::bail!("Only writer process can save database");
    }
    if !should_save() {
        return Ok(());
    }

    let path = Path::new(base_path);
    fs::create_dir_all(path)?;

    let mut school_data = HashMap::new();

    let idx_map = SCHOOL_INDICES.read().unwrap();
    let meta_map = SCHOOL_METADATA.read().unwrap();
    let em_map = SCHOOL_EMBEDDINGS.read().unwrap();

    for (school_id, (s_idx, t_idx)) in idx_map.iter() {
        let meta = meta_map.get(school_id).unwrap();
        let em = em_map.get(school_id).unwrap();

        let mut student_emb = Vec::new();
        let mut teacher_emb = Vec::new();

        for (id, m) in meta.iter() {
            if let Some(e) = em.get(id) {
                if m.role.to_lowercase() == "teacher" {
                    teacher_emb.push((e.clone(), *id));
                } else {
                    student_emb.push((e.clone(), *id));
                }
            }
        }

        school_data.insert(school_id.clone(), SchoolSerializable {
            student_embeddings: student_emb,
            teacher_embeddings: teacher_emb,
            metadata: meta.clone(),
        });
    }

    let data = SerializableData { version: 1, school_data };
    let tmp_path = path.join("data.bin.tmp");
    let final_path = path.join("data.bin");

    let bytes = bincode::serialize(&data)?;
    let file = File::create(&tmp_path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&bytes)?;
    writer.flush()?;
    fs::rename(tmp_path, final_path)?;

    Ok(())
}

/// Load all schools from disk
pub fn load_all(base_path: &str) -> Result<()> {
    if !is_writer() {
        return Ok(());
    }
    let path = Path::new(base_path);
    if !path.exists() {
        return Ok(());
    }

    let data_bytes = fs::read(path.join("data.bin"))?;
    let data: SerializableData = bincode::deserialize(&data_bytes)?;

    for (school_id, school) in data.school_data {
        init_school_indices(&school_id);

        let mut idx_map = SCHOOL_INDICES.write().unwrap();
        let (s_idx, t_idx) = idx_map.get_mut(&school_id).unwrap();
        *s_idx = Hnsw::new(16, 50_000, 100, 50, DistCosine);
        *t_idx = Hnsw::new(16, 5_000, 100, 50, DistCosine);

        for (emb, id) in &school.student_embeddings {
            s_idx.insert((&emb[..], *id));
        }
        for (emb, id) in &school.teacher_embeddings {
            t_idx.insert((&emb[..], *id));
        }

        let mut meta_map = SCHOOL_METADATA.write().unwrap();
        meta_map.insert(school_id.clone(), school.metadata.clone());

        let mut em_map = SCHOOL_EMBEDDINGS.write().unwrap();
        let ems = em_map.entry(school_id.clone()).or_default();
        ems.clear();
        for (emb, id) in school.student_embeddings.iter().chain(school.teacher_embeddings.iter()) {
            ems.insert(*id, emb.clone());
        }
    }

    // reset NEXT_ID to max across all schools
    let mut max_id = 0;
    let meta_map = SCHOOL_METADATA.read().unwrap();
    for school in meta_map.values() {
        if let Some(s) = school.keys().max() {
            max_id = max_id.max(*s);
        }
    }
    *NEXT_ID.lock().unwrap() = max_id + 1;

    Ok(())
}

/// Get metadata by ID within a school
pub fn get_metadata(school_id: &str, id: usize) -> Option<FaceMetadata> {
    let meta_map = SCHOOL_METADATA.read().unwrap();
    meta_map
        .get(school_id)
        .and_then(|m| m.get(&id))
        .cloned()
}

/// Total registered people in a school
pub fn get_total_faces(school_id: &str) -> usize {
    let meta_map = SCHOOL_METADATA.read().unwrap();
    meta_map
        .get(school_id)
        .map(|m| m.len())
        .unwrap_or(0)
}

/// Count students or teachers in a school
pub fn count_by_role(school_id: &str, role: &str) -> usize {
    let meta_map = SCHOOL_METADATA.read().unwrap();
    meta_map
        .get(school_id)
        .map(|m| {
            m.values()
                .filter(|meta| meta.role.eq_ignore_ascii_case(role))
                .count()
        })
        .unwrap_or(0)
}

/// Remove a face embedding (optional helper)
pub fn remove_face_embedding(school_id: &str, id: usize) -> Result<()> {
    if !is_writer() {
        anyhow::bail!("This process is not allowed to modify face database");
    }

    // Remove from metadata
    {
        let mut meta_map = SCHOOL_METADATA.write().unwrap();
        if let Some(meta) = meta_map.get_mut(school_id) {
            meta.remove(&id);
        }
    }

    // Remove from embeddings
    {
        let mut em_map = SCHOOL_EMBEDDINGS.write().unwrap();
        if let Some(embs) = em_map.get_mut(school_id) {
            embs.remove(&id);
        }
    }

    // Remove from indices
    {
        let mut idx_map = SCHOOL_INDICES.write().unwrap();
        if let Some((s_idx, t_idx)) = idx_map.get_mut(school_id) {
            // HNSW currently doesn't support delete natively, so we leave as-is or rebuild
        }
    }

    Ok(())
}

/// Get all embeddings for a person within a school
fn get_embeddings_for_person(
    school_id: &str,
    person_id: u64,
    role: &str
) -> Result<Vec<Vec<f32>>, String> {
    let meta_map = SCHOOL_METADATA.read().unwrap();
    let em_map = SCHOOL_EMBEDDINGS.read().unwrap();

    let school_meta = meta_map
        .get(school_id)
        .ok_or_else(|| "No metadata found for school".to_string())?;
    let school_em = em_map
        .get(school_id)
        .ok_or_else(|| "No embeddings found for school".to_string())?;

    let mut person_embeddings = Vec::new();
    for (id, meta) in school_meta.iter() {
        if meta.role.eq_ignore_ascii_case(role) && meta.person_id == person_id {
            if let Some(e) = school_em.get(id) {
                person_embeddings.push(e.clone());
            }
        }
    }

    if person_embeddings.is_empty() {
        return Err("No existing embeddings found for person".into());
    }

    Ok(person_embeddings)
}

pub fn batch_search(
    school_id: &str,
    embeddings: &[Vec<f32>],
    role: &str,
    k: usize
) -> Vec<Option<(usize, f32)>> {
    init_school_indices(school_id);

    let ef_search = 200;

    let idx_map = SCHOOL_INDICES.read().unwrap();
    let (s_idx, t_idx) = idx_map.get(school_id).unwrap();
    let idx = if role.to_lowercase() == "teacher" { t_idx } else { s_idx };

    embeddings
        .iter()
        .map(|emb| {
            idx.search(&emb[..], k, ef_search)
                .into_iter()
                .next()
                .map(|n| (n.d_id, 1.0 - n.distance))
        })
        .collect()
}
