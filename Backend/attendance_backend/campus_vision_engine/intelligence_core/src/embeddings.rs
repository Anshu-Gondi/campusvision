use crate::metadata::{ FaceMetadata, MetaHot };
use crate::utils::{ cosine_similarity, normalize };
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
use std::sync::{ Arc, atomic::{ AtomicUsize, Ordering } };

/// Writer guard
fn is_writer() -> bool {
    std::env
        ::var("FACE_DB_ROLE")
        .map(|v| v == "writer")
        .unwrap_or(false)
}

// HNSW <'static, f32, DistCosine> - fully Send + Sync for Python FFI
type IndexType = Hnsw<'static, f32, DistCosine>;

// Core data structure for a school's face database
struct SchoolShard {
    student_index: RwLock<IndexType>,
    teacher_index: RwLock<IndexType>,

    embeddings: RwLock<Vec<Vec<f32>>>,
    metadata_hot: RwLock<Vec<MetaHot>>,
    metadata_full: RwLock<Vec<FaceMetadata>>,

    cache: Mutex<LruCache<String, Vec<f32>>>,
    delete_count: Mutex<usize>,

    next_id: AtomicUsize,
}

static SCHOOL_SHARDS: Lazy<RwLock<HashMap<String, Arc<SchoolShard>>>> = Lazy::new(||
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
fn get_or_create_shard(school_id: &str) -> Arc<SchoolShard> {
    {
        let map = SCHOOL_SHARDS.read().unwrap();
        if let Some(shard) = map.get(school_id) {
            return Arc::clone(shard);
        }
    }

    let mut map = SCHOOL_SHARDS.write().unwrap();

    Arc::clone(
        map.entry(school_id.to_string()).or_insert_with(|| {
            Arc::new(SchoolShard {
                student_index: RwLock::new(Hnsw::new(16, 50_000, 100, 50, DistCosine)),
                teacher_index: RwLock::new(Hnsw::new(16, 5_000, 100, 50, DistCosine)),

                embeddings: RwLock::new(Vec::new()),
                metadata_hot: RwLock::new(Vec::new()),
                metadata_full: RwLock::new(Vec::new()),

                cache: Mutex::new(LruCache::new(NonZeroUsize::new(1000).unwrap())),
                delete_count: Mutex::new(0),
                next_id: AtomicUsize::new(0),
            })
        })
    )
}

/// Add embedding for a school
pub fn add_face_embedding(
    school_id: &str,
    mut embedding: Vec<f32>, // 👈 mutable for normalization
    name: String,
    person_id: u64,
    roll_no: String,
    role: String
) -> Result<usize> {
    if !is_writer() {
        anyhow::bail!("Not allowed");
    }

    let shard = get_or_create_shard(school_id);
    let is_teacher = role.eq_ignore_ascii_case("teacher");

    // 🔥 atomic id (NO LOCK)
    let id = shard.next_id.fetch_add(1, Ordering::Relaxed);

    // 🔥 normalize ONCE (big win)
    normalize(&mut embedding);

    // 🔥 insert into HNSW
    if is_teacher {
        shard.teacher_index
            .write()
            .unwrap()
            .insert((&embedding[..], id));
    } else {
        shard.student_index
            .write()
            .unwrap()
            .insert((&embedding[..], id));
    }

    // 🔥 embeddings (dense vector)
    {
        let mut em = shard.embeddings.write().unwrap();
        if em.len() == id {
            em.push(embedding.clone());
        } else {
            em[id] = embedding.clone();
        }
    }

    // 🔥 hot metadata (FAST PATH)
    {
        let mut hot = shard.metadata_hot.write().unwrap();
        hot.push(MetaHot {
            deleted: false,
            role: if is_teacher {
                1
            } else {
                0
            },
            person_id,
        });
    }

    // 🔥 full metadata (cold path)
    {
        let mut full = shard.metadata_full.write().unwrap();
        full.push(FaceMetadata {
            id: id.to_string(),
            name,
            person_id,
            roll_no,
            role,
            reliability: None,
            deleted: false,
        });
    }

    // cache
    shard.cache.lock().unwrap().put(format!("{}:{}", role.to_lowercase(), person_id), embedding);

    Ok(id)
}

/// Search within a school & role
pub fn search_in_role(
    school_id: &str,
    embedding: &[f32],
    role: &str,
    k: usize
) -> Vec<(usize, f32)> {
    let shard = get_or_create_shard(school_id);
    let is_teacher = role.eq_ignore_ascii_case("teacher");

    let idx = if is_teacher {
        shard.teacher_index.read().unwrap()
    } else {
        shard.student_index.read().unwrap()
    };

    let hot = shard.metadata_hot.read().unwrap();

    idx.search(embedding, k * 3, 100)
        .into_iter()
        .filter_map(|n| {
            let m = &hot[n.d_id];

            if !m.deleted {
                Some((n.d_id, 1.0 - n.distance))
            } else {
                None
            }
        })
        .take(k)
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

    // 🔥 USE SHARDS (NOT GLOBALS)
    let shards: Vec<(String, Arc<SchoolShard>)> = {
        let map = SCHOOL_SHARDS.read().unwrap();
        map.iter()
            .map(|(k, v)| (k.clone(), Arc::clone(v)))
            .collect()
    };

    for (school_id, shard) in shards {
        let full = shard.metadata_full.read().unwrap();
        let hot = shard.metadata_hot.read().unwrap();
        let em = shard.embeddings.read().unwrap();

        let mut student_emb = Vec::new();
        let mut teacher_emb = Vec::new();
        let mut metadata = HashMap::new();

        for (id, m) in full.iter().enumerate() {
            if hot[id].deleted {
                continue;
            }

            let e = &em[id];

            if hot[id].role == 1 {
                teacher_emb.push((e.clone(), id));
            } else {
                student_emb.push((e.clone(), id));
            }

            metadata.insert(id, m.clone());
        }

        school_data.insert(school_id.clone(), SchoolSerializable {
            student_embeddings: student_emb,
            teacher_embeddings: teacher_emb,
            metadata,
        });
    }

    let data = SerializableData {
        version: 1,
        school_data,
    };

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

    let mut shard_max_id = 0;

    for (school_id, school) in data.school_data {
        let shard = get_or_create_shard(&school_id);

        let mut local_max_id = 0;

        // 🔹 RESET indices
        let mut student_idx = Hnsw::new(16, 50_000, 100, 50, DistCosine);
        let mut teacher_idx = Hnsw::new(16, 5_000, 100, 50, DistCosine);

        // 🔹 prepare vectors
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        let mut metadata_hot: Vec<MetaHot> = Vec::new();
        let mut metadata_full: Vec<FaceMetadata> = Vec::new();

        // 🔹 rebuild from saved data
        for (id, meta) in &school.metadata {
            let role_flag = if meta.role.eq_ignore_ascii_case("teacher") { 1 } else { 0 };

            // find embedding
            let emb = school.student_embeddings
                .iter()
                .chain(school.teacher_embeddings.iter())
                .find(|(_, eid)| eid == id)
                .map(|(e, _)| e.clone());

            if let Some(e) = emb {
                let id_usize = *id;

                // ensure vec size
                if embeddings.len() <= id_usize {
                    embeddings.resize(id_usize + 1, vec![]);
                    metadata_hot.resize(id_usize + 1, MetaHot {
                        deleted: true,
                        role: 0,
                        person_id: 0,
                    });
                    metadata_full.resize(id_usize + 1, meta.clone());
                }

                embeddings[id_usize] = e.clone();

                metadata_hot[id_usize] = MetaHot {
                    deleted: false,
                    role: role_flag,
                    person_id: meta.person_id,
                };

                metadata_full[id_usize] = meta.clone();

                // insert into index
                if role_flag == 1 {
                    teacher_idx.insert((&e[..], id_usize));
                } else {
                    student_idx.insert((&e[..], id_usize));
                }

                local_max_id = local_max_id.max(id_usize);
            }
        }

        // 🔹 swap everything
        *shard.student_index.write().unwrap() = student_idx;
        *shard.teacher_index.write().unwrap() = teacher_idx;

        *shard.embeddings.write().unwrap() = embeddings;
        *shard.metadata_hot.write().unwrap() = metadata_hot;
        *shard.metadata_full.write().unwrap() = metadata_full;

        shard.next_id.store(local_max_id + 1, Ordering::Relaxed);
    }

    Ok(())
}

/// Get metadata by ID within a school
pub fn get_metadata(school_id: &str, id: usize) -> Option<FaceMetadata> {
    let shard = get_or_create_shard(school_id);

    let meta = shard.metadata_full.read().unwrap();
    meta.get(id).cloned()
}

/// Total registered people in a school
pub fn get_total_faces(school_id: &str) -> usize {
    let shard = get_or_create_shard(school_id);

    let meta = shard.metadata_hot.read().unwrap();
    meta.iter()
        .filter(|m| !m.deleted)
        .count()
}

/// Count students or teachers in a school
pub fn count_by_role(school_id: &str, role: &str) -> usize {
    let shard = get_or_create_shard(school_id);

    let meta = shard.metadata_hot.read().unwrap();

    let role_flag = if role.eq_ignore_ascii_case("teacher") { 1 } else { 0 };

    meta.iter()
        .filter(|m| !m.deleted && m.role == role_flag)
        .count()
}

/// Remove a face embedding (optional helper)
pub fn remove_face_embedding(school_id: &str, id: usize) -> Result<()> {
    if !is_writer() {
        anyhow::bail!("Not allowed");
    }

    let shard = get_or_create_shard(school_id);

    let mut meta = shard.metadata_hot.write().unwrap();

    if let Some(m) = meta.get_mut(&id) {
        if !m.deleted {
            m.deleted = true;

            let mut dc = shard.delete_count.lock().unwrap();
            *dc += 1;
        }
    }

    rebuild_index_if_needed(school_id);

    Ok(())
}

/// Get all embeddings for a person within a school
fn get_embeddings_for_person(
    school_id: &str,
    person_id: u64,
    role: &str
) -> Result<Vec<Vec<f32>>, String> {
    let shard = get_or_create_shard(school_id);

    let meta = shard.metadata_hot.read().unwrap();
    let em = shard.embeddings.read().unwrap();

    let role_flag = if role.eq_ignore_ascii_case("teacher") { 1 } else { 0 };

    let mut person_embeddings = Vec::new();

    for (id, m) in meta.iter().enumerate() {
        if m.deleted {
            continue;
        }

        if m.person_id == person_id && m.role == role_flag {
            person_embeddings.push(em[id].clone());
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
    let shard = get_or_create_shard(school_id);

    let is_teacher = role.eq_ignore_ascii_case("teacher");

    let idx = if is_teacher {
        shard.teacher_index.read().unwrap()
    } else {
        shard.student_index.read().unwrap()
    };

    let hot = shard.metadata_hot.read().unwrap();

    embeddings
        .iter()
        .map(|emb| {
            idx.search(&emb[..], k * 3, 200)
                .into_iter()
                .find_map(|n| {
                    let m = &hot[n.d_id];

                    if !m.deleted {
                        Some((n.d_id, 1.0 - n.distance))
                    } else {
                        None
                    }
                })
        })
        .collect()
}

pub fn rebuild_index_if_needed(school_id: &str) {
    if !is_writer() {
        return;
    }

    const DELETE_THRESHOLD: usize = 50;

    let shard = get_or_create_shard(school_id);

    // check delete count
    {
        let mut dc = shard.delete_count.lock().unwrap();
        if *dc < DELETE_THRESHOLD {
            return;
        }
        *dc = 0;
    }

    // snapshot
    let meta = shard.metadata_hot.read().unwrap().clone();
    let em = shard.embeddings.read().unwrap().clone();

    // rebuild
    let mut new_s = Hnsw::new(16, 50_000, 100, 50, DistCosine);
    let mut new_t = Hnsw::new(16, 5_000, 100, 50, DistCosine);

    for (id, m) in meta.iter().enumerate() {
        if m.deleted {
            continue;
        }

        let e = &em[id];

        if m.role == 1 {
            new_t.insert((&e[..], id));
        } else {
            new_s.insert((&e[..], id));
        }
    }

    // swap
    *shard.student_index.write().unwrap() = new_s;
    *shard.teacher_index.write().unwrap() = new_t;
}
