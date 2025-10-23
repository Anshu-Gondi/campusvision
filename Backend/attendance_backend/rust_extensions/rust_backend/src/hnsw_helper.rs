use anyhow::Result;
use hnsw_rs::prelude::*;
use once_cell::sync::Lazy;
use std::sync::Mutex;

static HNSW_INDEX: Lazy<Mutex<Hnsw<f32, DistCosine>>> = Lazy::new(|| {
    Mutex::new(Hnsw::new(16, 100, 200, 50, DistCosine))
});

pub fn add_embedding(embedding: &[f32]) -> Result<()> {
    let index = HNSW_INDEX.lock().unwrap();
    let id = index.get_nb_point() as usize;
    index.insert((&embedding.to_vec(), id));
    Ok(())
}

pub fn query_embedding(embedding: &[f32], k: usize) -> Vec<usize> {
    let index = HNSW_INDEX.lock().unwrap();
    index
        .search(&embedding.to_vec(), k, 50)
        .into_iter()
        .map(|r| r.d_id)
        .collect()
}
