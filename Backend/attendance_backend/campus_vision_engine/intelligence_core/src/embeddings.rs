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

// Internal data structure for a shard (school)
struct ShardData {
    embeddings: Vec<Vec<f32>>,
    metadata_hot: Vec<MetaHot>,
    metadata_full: Vec<FaceMetadata>,
    next_id: usize,
}
// Core data structure for a school's face database
struct SchoolShard {
    data: RwLock<ShardData>,

    student_index: RwLock<IndexType>,
    teacher_index: RwLock<IndexType>,

    cache: Mutex<LruCache<String, Vec<f32>>>,
    delete_count: AtomicUsize,
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

                cache: Mutex::new(LruCache::new(NonZeroUsize::new(1000).unwrap())),
                data: RwLock::new(ShardData {
                    embeddings: Vec::new(),
                    metadata_hot: Vec::new(),
                    metadata_full: Vec::new(),
                    next_id: 0,
                }),
                delete_count: AtomicUsize::new(0),
            })
        })
    )
}

/// Add embedding for a school
pub fn add_face_embedding(
    school_id: &str,
    mut embedding: Vec<f32>,
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
    let role_key: &str = if is_teacher { "teacher" } else { "student" };

    normalize(&mut embedding);

    // ✅ SINGLE ATOMIC DATA UPDATE
    let id = {
        let mut d = shard.data.write().unwrap();

        let id = d.next_id;
        d.next_id += 1;

        d.embeddings.push(embedding.clone());

        d.metadata_hot.push(MetaHot {
            deleted: false,
            role: if is_teacher {
                1
            } else {
                0
            },
            person_id,
        });

        d.metadata_full.push(FaceMetadata {
            id: id.to_string(),
            name,
            person_id,
            roll_no,
            role,
            reliability: None,
            deleted: false,
        });

        id
    };

    // ✅ HNSW AFTER DATA (safe)
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

    // ✅ cache
    shard.cache.lock().unwrap().put(format!("{}:{}", role_key, person_id), embedding);

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

    // 🔹 Phase 1: search index ONLY
    let raw_results = {
        let idx = if is_teacher {
            shard.teacher_index.read().unwrap()
        } else {
            shard.student_index.read().unwrap()
        };

        idx.search(embedding, k * 3, 100)
            .into_iter()
            .map(|n| (n.d_id, 1.0 - n.distance))
            .collect::<Vec<_>>()
    }; // index lock released here

    // 🔹 Phase 2: validate against data safely
    let d = shard.data.read().unwrap();

    raw_results
        .into_iter()
        .filter_map(|(id, score)| {
            d.metadata_hot
                .get(id) // ✅ SAFE (no panic)
                .filter(|m| !m.deleted) // ✅ skip deleted
                .map(|_| (id, score))
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
        let d = shard.data.read().unwrap();

        let mut student_emb = Vec::new();
        let mut teacher_emb = Vec::new();
        let mut metadata = HashMap::new();

        for (id, m) in d.metadata_full.iter().enumerate() {
            if d.metadata_hot[id].deleted {
                continue;
            }

            let e = &d.embeddings[id];

            if d.metadata_hot[id].role == 1 {
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

        // 🔹 build lookup map
        let mut emb_map: HashMap<usize, Vec<f32>> = HashMap::new();

        for (e, id) in school.student_embeddings.iter().chain(school.teacher_embeddings.iter()) {
            emb_map.insert(*id, e.clone());
        }

        // 🔹 rebuild from saved data
        for (id, meta) in &school.metadata {
            let role_flag = if meta.role.eq_ignore_ascii_case("teacher") { 1 } else { 0 };

            if let Some(mut e) = emb_map.get(id).cloned() {
                let id_usize = *id;

                normalize(&mut e);

                // ensure capacity
                if embeddings.len() <= id_usize {
                    embeddings.resize(id_usize + 1, vec![]);

                    metadata_hot.resize(id_usize + 1, MetaHot {
                        deleted: true,
                        role: 0,
                        person_id: 0,
                    });

                    metadata_full.resize_with(id_usize + 1, || FaceMetadata {
                        id: "".into(),
                        name: "".into(),
                        person_id: 0,
                        roll_no: "".into(),
                        role: "".into(),
                        reliability: None,
                        deleted: true,
                    });
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

        // ✅ STEP 1: swap indices (independent lifecycle — fine)
        *shard.student_index.write().unwrap() = student_idx;
        *shard.teacher_index.write().unwrap() = teacher_idx;

        // ✅ STEP 2: SINGLE ATOMIC DATA SWAP (this is the real fix)
        {
            let mut d = shard.data.write().unwrap();

            d.embeddings = embeddings;
            d.metadata_hot = metadata_hot;
            d.metadata_full = metadata_full;
            d.next_id = local_max_id + 1;
        }
    }

    Ok(())
}

/// Get metadata by ID within a school
pub fn get_metadata(school_id: &str, id: usize) -> Option<FaceMetadata> {
    let shard = get_or_create_shard(school_id);

    let d = shard.data.read().unwrap();

    d.metadata_full.get(id).cloned()
}

/// Total registered people in a school
pub fn get_total_faces(school_id: &str) -> usize {
    let shard = get_or_create_shard(school_id);

    let d = shard.data.read().unwrap();

    d.metadata_hot
        .iter()
        .filter(|m| !m.deleted)
        .count()
}

/// Count students or teachers in a school
pub fn count_by_role(school_id: &str, role: &str) -> usize {
    let shard = get_or_create_shard(school_id);

    let role_flag = if role.eq_ignore_ascii_case("teacher") { 1 } else { 0 };

    let d = shard.data.read().unwrap();

    d.metadata_hot
        .iter()
        .filter(|m| !m.deleted && m.role == role_flag)
        .count()
}

/// Remove a face embedding (optional helper)
pub fn remove_face_embedding(school_id: &str, id: usize) -> Result<()> {
    if !is_writer() {
        anyhow::bail!("Not allowed");
    }

    let shard = get_or_create_shard(school_id);

    // ✅ SINGLE WRITE LOCK (atomic update)
    let deleted = {
        let mut d = shard.data.write().unwrap();

        if let Some(m) = d.metadata_hot.get_mut(id) {
            if !m.deleted {
                m.deleted = true;

                if let Some(f) = d.metadata_full.get_mut(id) {
                    f.deleted = true;
                }

                true
            } else {
                false
            }
        } else {
            false
        }
    }; // 🔥 lock released here

    // ✅ atomic counter (NO mutex anymore)
    if deleted {
        shard.delete_count.fetch_add(1, Ordering::Relaxed);
    }

    // ✅ rebuild decision outside lock
    rebuild_index_if_needed(school_id);

    Ok(())
}

/// Get all embeddings for a person within a school
pub fn get_embeddings_for_person(
    school_id: &str,
    person_id: u64,
    role: &str
) -> Result<Vec<Vec<f32>>, String> {
    let shard = get_or_create_shard(school_id);

    let role_flag = if role.eq_ignore_ascii_case("teacher") { 1 } else { 0 };

    // ✅ single consistent snapshot
    let d = shard.data.read().unwrap();

    let mut person_embeddings = Vec::new();

    for (id, m) in d.metadata_hot.iter().enumerate() {
        if m.deleted {
            continue;
        }

        if m.person_id == person_id && m.role == role_flag {
            // safe because embeddings + metadata are aligned
            if let Some(e) = d.embeddings.get(id) {
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
    let shard = get_or_create_shard(school_id);
    let is_teacher = role.eq_ignore_ascii_case("teacher");

    // 🔹 Phase 1: read all data under single snapshot
    let d = shard.data.read().unwrap();
    let idx = if is_teacher {
        shard.teacher_index.read().unwrap()
    } else {
        shard.student_index.read().unwrap()
    };

    embeddings
        .iter()
        .map(|emb| {
            idx.search(&emb[..], k * 3, 200)
                .into_iter()
                .find_map(|n| {
                    // ✅ safe access, consistent
                    d.metadata_hot.get(n.d_id).and_then(|m| {
                        if !m.deleted { Some((n.d_id, 1.0 - n.distance)) } else { None }
                    })
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

    // 🔹 check and reset delete count
    {
        let mut dc = shard.delete_count.lock().unwrap();
        if *dc < DELETE_THRESHOLD {
            return;
        }
        *dc = 0;
    }

    // 🔹 snapshot current data
    let d = shard.data.read().unwrap();
    let meta = d.metadata_hot.clone();
    let em = d.embeddings.clone();

    // 🔹 rebuild HNSW indices
    let mut new_student_idx = Hnsw::new(16, 50_000, 100, 50, DistCosine);
    let mut new_teacher_idx = Hnsw::new(16, 5_000, 100, 50, DistCosine);

    for (id, m) in meta.iter().enumerate() {
        if m.deleted {
            continue;
        }

        let e = &em[id];

        if m.role == 1 {
            new_teacher_idx.insert((&e[..], id));
        } else {
            new_student_idx.insert((&e[..], id));
        }
    }

    // 🔹 swap indices atomically
    *shard.student_index.write().unwrap() = new_student_idx;
    *shard.teacher_index.write().unwrap() = new_teacher_idx;
}
