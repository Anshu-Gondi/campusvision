use anyhow::Result;
use hnsw_rs::prelude::*;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

/// Metadata stored for each face
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FaceMetadata {
    pub id: usize,
    pub name: String,
    pub roll_no: String,
    pub role: String, // "student" or "teacher"
}

// Hnsw<'static, f32, DistCosine> – fully Send + Sync for Python
type IndexType = Hnsw<'static, f32, DistCosine>;

static STUDENT_INDEX: Lazy<Mutex<IndexType>> = Lazy::new(|| {
    // M = 16, max_elements = 50_000, ef_construction = 200, max_nb_links = 50
    Mutex::new(Hnsw::new(16, 50_000, 200, 50, DistCosine))
});

static TEACHER_INDEX: Lazy<Mutex<IndexType>> = Lazy::new(|| {
    Mutex::new(Hnsw::new(16, 5_000, 200, 50, DistCosine))
});

static METADATA: Lazy<Mutex<HashMap<usize, FaceMetadata>>> = Lazy::new(|| Mutex::new(HashMap::new()));

// Store actual embeddings keyed by id so we can save and rebuild the graph on load
static EMBEDDINGS: Lazy<Mutex<HashMap<usize, Vec<f32>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Add a face embedding with name, roll_no, and role
pub fn add_embedding(
    embedding: Vec<f32>,
    name: String,
    roll_no: String,
    role: String,
) -> Result<usize> {
    let is_teacher = role == "teacher";
    // lock index for insertion
    let idx = if is_teacher {
        TEACHER_INDEX.lock().unwrap()
    } else {
        STUDENT_INDEX.lock().unwrap()
    };

    // next id is current number of points
    let base_id = idx.get_nb_point() as usize;
    let id = if is_teacher { 1_000_000 + base_id } else { base_id };

    // insert embedding into index (borrow slice)
    idx.insert((&embedding[..], id));

    // persist metadata + embedding in memory maps
    {
        let mut meta = METADATA.lock().unwrap();
        meta.insert(
            id,
            FaceMetadata {
                id,
                name,
                roll_no,
                role: if is_teacher { "teacher".to_string() } else { "student".to_string() },
            },
        );
    }
    {
        let mut em = EMBEDDINGS.lock().unwrap();
        em.insert(id, embedding);
    }

    Ok(id)
}

/// Search only within student or teacher index
pub fn search_in_role(embedding: &[f32], role: &str, k: usize) -> Vec<(usize, f32)> {
    // use higher ef for search for better recall; we pass ef_search=200
    let ef_search = 200usize;

    let idx = if role == "teacher" {
        TEACHER_INDEX.lock().unwrap()
    } else {
        STUDENT_INDEX.lock().unwrap()
    };

    idx.search(embedding, k, ef_search)
        .into_iter()
        .map(|n| (n.d_id, 1.0 - n.distance)) // (id, similarity)
        .collect()
}

/// Get metadata by ID
pub fn get_metadata(id: usize) -> Option<FaceMetadata> {
    METADATA.lock().unwrap().get(&id).cloned()
}

/// Total registered people
pub fn get_total_faces() -> usize {
    METADATA.lock().unwrap().len()
}

/// Count students or teachers
pub fn count_by_role(role: &str) -> usize {
    METADATA
        .lock()
        .unwrap()
        .values()
        .filter(|m| m.role == role)
        .count()
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

    let meta = METADATA.lock().unwrap();
    let em = EMBEDDINGS.lock().unwrap();

    let mut student_embs = Vec::new();
    let mut teacher_embs = Vec::new();

    for (id, m) in meta.iter() {
        if let Some(embedding) = em.get(id) {
            if m.role == "student" {
                student_embs.push((embedding.clone(), *id));
            } else {
                teacher_embs.push((embedding.clone(), *id));
            }
        } else {
            // skip entries without saved embeddings (shouldn't usually happen)
        }
    }

    let data = SerializableData {
        student_embeddings: student_embs,
        teacher_embeddings: teacher_embs,
        metadata: meta.clone(),
    };

    fs::write(p.join("data.bin"), bincode::serialize(&data)?)?;
    Ok(())
}

/// Load from disk & REBUILD HNSW
pub fn load_all(base_path: &str) -> Result<()> {
    let p = Path::new(base_path);
    if !p.exists() {
        return Ok(());
    }

    if let Ok(data_bytes) = fs::read(p.join("data.bin")) {
        if let Ok(data) = bincode::deserialize::<SerializableData>(&data_bytes) {
            // restore metadata and embeddings
            {
                let mut meta = METADATA.lock().unwrap();
                *meta = data.metadata;
            }
            {
                let mut em = EMBEDDINGS.lock().unwrap();
                em.clear();
                for (emb, id) in data.student_embeddings.iter().chain(data.teacher_embeddings.iter()) {
                    em.insert(*id, emb.clone());
                }
            }

            // Rebuild indices from saved embeddings
            {
                let mut s_idx = STUDENT_INDEX.lock().unwrap();
                let mut t_idx = TEACHER_INDEX.lock().unwrap();
                *s_idx = Hnsw::new(16, 50_000, 200, 50, DistCosine);
                *t_idx = Hnsw::new(16, 5_000, 200, 50, DistCosine);

                // insert student embeddings
                for (emb, id) in data.student_embeddings {
                    s_idx.insert((&emb[..], id));
                }
                // insert teacher embeddings
                for (emb, id) in data.teacher_embeddings {
                    t_idx.insert((&emb[..], id));
                }
            }
        }
    }

    Ok(())
}
