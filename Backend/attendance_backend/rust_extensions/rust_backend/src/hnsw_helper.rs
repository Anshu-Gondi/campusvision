use anyhow::Result;
use hnsw_rs::prelude::*;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::RwLock;
use crate::utils::cosine_similarity;

/// Metadata stored for each face
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FaceMetadata {
    pub id: String,
    pub name: String,
    pub person_id: u64,
    pub roll_no: String,
    pub role: String, // e.g., "student", "teacher"
    pub reliability: Option<f32>,
}

// HNSW <'static, f32, DistCosine> - fully Send + Sync for Python FFI
type IndexType = Hnsw<'static, f32, DistCosine>;

static STUDENT_INDEX: Lazy<RwLock<IndexType>> = Lazy::new(|| {
    // M = 16, max_elements = 50_000, ef_construction = 100, max_nb_links = 50
    RwLock::new(Hnsw::new(16, 50_000, 100, 50, DistCosine))
});

static TEACHER_INDEX: Lazy<RwLock<IndexType>> =
    Lazy::new(|| RwLock::new(Hnsw::new(16, 5_000, 100, 50, DistCosine)));

static METADATA: Lazy<RwLock<HashMap<usize, FaceMetadata>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Stores raw embeddings by ID (used for debugging / persistence)
static EMBEDDINGS: Lazy<RwLock<HashMap<usize, Vec<f32>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn add_face_embedding(
    embedding: Vec<f32>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String,
) -> Result<usize> {
    let role = role.to_lowercase();
    let is_teacher = role == "teacher";

    // Pick index
    let index = if is_teacher {
        TEACHER_INDEX.write().unwrap()
    } else {
        STUDENT_INDEX.write().unwrap()
    };

    // Generate ID (teacher IDs are offset to avoid collision)
    let local_id = index.get_nb_point() as usize;
    let id = if is_teacher {
        1_000_000 + local_id
    } else {
        local_id
    };

    // Insert embedding into HNSW (NO `?`, insert returns ())
    index.insert((&embedding[..], id));

    // Store metadata
    {
        let mut meta = METADATA.write().unwrap();
        meta.insert(
            id,
            FaceMetadata {
                id: id.to_string(),
                name,
                person_id,
                roll_no,
                role,
                reliability: None,
            },
        );
    }

    // Store embedding (for persistence / reload / debug)
    {
        let mut em = EMBEDDINGS.write().unwrap();
        em.insert(id, embedding);
    }

    Ok(id)
}

fn get_embeddings_for_person(
    person_id: u64,
    role: &str,
) -> Result<Vec<Vec<f32>>, String> {
    let meta = METADATA.read().unwrap();
    let embs = EMBEDDINGS.read().unwrap();

    let mut result = Vec::new();

    for (id, m) in meta.iter() {
        if m.role == role && m.id.parse::<u64>().ok() == Some(person_id) {
            if let Some(e) = embs.get(id) {
                result.push(e.clone());
            }
        }
    }

    if result.is_empty() {
        return Err("No existing embeddings found for person".into());
    }

    Ok(result)
}

pub fn can_reenroll(
    embedding: &Vec<f32>,
    person_id: u64,
    role: &str,
) -> Result<bool, String> {
    let embeddings = get_embeddings_for_person(person_id, role)?;

    let max_sim = embeddings
        .iter()
        .map(|e| cosine_similarity(e, embedding))
        .fold(0.0, f32::max);

    Ok(max_sim >= 0.65)
}

/// Search only within student or teacher index based on role
pub fn search_in_role(embedding: &[f32], role: &str, k: usize) -> Vec<(usize, f32)> {
    let ef_search = 100usize; // tuned for speed/accuracy tradeoff

    let idx = if role.to_lowercase() == "teacher" {
        TEACHER_INDEX.read().unwrap()
    } else {
        STUDENT_INDEX.read().unwrap()
    };

    let hits = idx.search(embedding, k, ef_search);

    let mut results = Vec::with_capacity(k);
    for n in hits {
        results.push((n.d_id, 1.0 - n.distance)); // id and similarity
    }
    results
}

/// Get metadata by ID
pub fn get_metadata(id: usize) -> Option<FaceMetadata> {
    METADATA.read().unwrap().get(&id).cloned()
}

/// Total registered people
pub fn get_total_faces() -> usize {
    METADATA.read().unwrap().len()
}

/// Count students or teachers
pub fn count_by_role(role: &str) -> usize {
    METADATA
        .read()
        .unwrap()
        .values()
        .filter(|m| m.role.to_lowercase() == role.to_lowercase())
        .count()
}

/// batch search embeddings by role
pub fn batch_search(
    embeddings: &[Vec<f32>],
    role: &str,
    k: usize,
) -> Vec<Option<(usize, f32)>> {
    let ef_search = 200;

    let idx = if role == "teacher" {
        TEACHER_INDEX.read().unwrap()
    } else {
        STUDENT_INDEX.read().unwrap()
    };

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


/// Save embeddings + metadata (rebuild HNSW on load)
#[derive(Serialize, Deserialize)]
struct SerializableData {
    student_embeddings: Vec<(Vec<f32>, usize)>,
    teacher_embeddings: Vec<(Vec<f32>, usize)>,
    metadata: HashMap<usize, FaceMetadata>,
}

pub fn save_all(base_path: &str) -> Result<()> {
    let p = Path::new(base_path);
    fs::create_dir_all(p)?;

    let meta = METADATA.read().unwrap();
    let em = EMBEDDINGS.read().unwrap();

    let mut student_emb = Vec::new();
    let mut teacher_emb = Vec::new();

    for (id, m) in meta.iter() {
        if let Some(embedding) = em.get(id) {
            if m.role.to_lowercase() == "teacher" {
                teacher_emb.push((embedding.clone(), *id));
            } else {
                student_emb.push((embedding.clone(), *id));
            }
        }
    }

    let data = SerializableData {
        student_embeddings: student_emb,
        teacher_embeddings: teacher_emb,
        metadata: meta.clone(),
    };

    fs::write(p.join("data.bin"), bincode::serialize(&data)?)?;
    Ok(())
}

/// Load from disk & REBUILD HNSW in parallel
pub fn load_all(base_path: &str) -> Result<()> {
    let p = Path::new(base_path);
    if !p.exists() {
        return Ok(());
    }

    if let Ok(data_bytes) = fs::read(p.join("data.bin")) {
        if let Ok(data) = bincode::deserialize::<SerializableData>(&data_bytes) {
            {
                let mut meta = METADATA.write().unwrap();
                *meta = data.metadata;
            }

            {
                let mut em = EMBEDDINGS.write().unwrap();
                em.clear();
                for (emb, id) in data
                    .student_embeddings
                    .iter()
                    .chain(data.teacher_embeddings.iter())
                {
                    em.insert(*id, emb.clone());
                }
            }

            // Rebuild indices in parallel
            {
                let mut s_idx = STUDENT_INDEX.write().unwrap();
                let mut t_idx = TEACHER_INDEX.write().unwrap();

                *s_idx = Hnsw::new(16, 50_000, 100, 50, DistCosine);
                *t_idx = Hnsw::new(16, 5_000, 100, 50, DistCosine);

                data.student_embeddings.par_iter().for_each(|(emb, id)| {
                    s_idx.insert((&emb[..], *id));
                });

                data.teacher_embeddings.par_iter().for_each(|(emb, id)| {
                    t_idx.insert((&emb[..], *id));
                });
            }
        }
    }

    Ok(())
}
