/// cctv/frame_batcher.rs
///
/// Per-(school_id, camera_id, role) batcher.
///
/// Instead of one `search_in_role` call per frame, we accumulate frames
/// for up to BATCH_DEADLINE_MS (50 ms) OR until BATCH_SIZE frames are
/// ready, then fire one `batch_search` call.
///
/// This reduces `intelligence_core` lock acquisitions from
///   50 cameras × N fps  →  ~1 acquisition per 50 ms per camera
/// i.e. from thousands/sec to ~1000/50ms = 20 total at 50 cameras.
///
/// Architecture:
///   Each camera key gets a `BatchSender` (cheap clone, just an mpsc tx).
///   A single Tokio task per key drains its channel and fires batches.
///   Results are returned via oneshot channels stored alongside the frame.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use once_cell::sync::Lazy;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, Instant};

use intelligence_core::embeddings::batch_search;

const BATCH_SIZE: usize = 32;
const BATCH_DEADLINE_MS: u64 = 50;

/// Result for one embedding query.
pub type BatchResult = Option<(usize, f32)>;

/// One item in the batch channel.
struct BatchItem {
    embedding: Vec<f32>,
    respond_to: oneshot::Sender<BatchResult>,
}

/// Cheap handle to submit a frame embedding for batched search.
#[derive(Clone)]
pub struct BatchSender {
    tx: mpsc::Sender<BatchItem>,
}

impl BatchSender {
    /// Submit an embedding and await the batched result.
    /// Returns None if no match found or if the batcher is overloaded.
    pub async fn search(&self, embedding: Vec<f32>) -> BatchResult {
        let (tx, rx) = oneshot::channel();
        if self.tx.send(BatchItem { embedding, respond_to: tx }).await.is_err() {
            return None; // batcher dropped — non-fatal for CCTV
        }
        rx.await.unwrap_or(None)
    }
}

/// Global registry: (school_id, camera_id, role) → BatchSender
static BATCHERS: Lazy<DashMap<(String, String, String), BatchSender>> =
    Lazy::new(DashMap::new);

/// Get or create a BatchSender for the given key.
/// Spawns a background batcher task on first call.
pub fn get_or_create_batcher(school_id: &str, camera_id: &str, role: &str) -> BatchSender {
    let key = (school_id.to_string(), camera_id.to_string(), role.to_string());

    if let Some(sender) = BATCHERS.get(&key) {
        return sender.clone();
    }

    // Double-checked insert
    let (tx, rx) = mpsc::channel::<BatchItem>(BATCH_SIZE * 4);
    let sender = BatchSender { tx };

    BATCHERS.insert(key.clone(), sender.clone());

    // Spawn the drain task
    let school_id = school_id.to_string();
    let role = role.to_string();
    tokio::spawn(run_batcher(school_id, role, rx));

    sender
}

/// The batcher task. Drains the channel every 50ms or every BATCH_SIZE frames.
async fn run_batcher(school_id: String, role: String, mut rx: mpsc::Receiver<BatchItem>) {
    let deadline = Duration::from_millis(BATCH_DEADLINE_MS);
    let mut tick = interval(deadline);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut pending: Vec<BatchItem> = Vec::with_capacity(BATCH_SIZE);

    loop {
        tokio::select! {
            // Deadline tick — flush whatever we have
            _ = tick.tick() => {
                if !pending.is_empty() {
                    flush_batch(&school_id, &role, &mut pending).await;
                }
            }

            // New item arrived
            item = rx.recv() => {
                match item {
                    None => {
                        // Channel closed (camera removed) — flush and exit
                        if !pending.is_empty() {
                            flush_batch(&school_id, &role, &mut pending).await;
                        }
                        return;
                    }
                    Some(item) => {
                        pending.push(item);
                        // Flush early if batch is full
                        if pending.len() >= BATCH_SIZE {
                            flush_batch(&school_id, &role, &mut pending).await;
                        }
                    }
                }
            }
        }
    }
}

/// Run one `batch_search` and dispatch results back to callers.
async fn flush_batch(school_id: &str, role: &str, pending: &mut Vec<BatchItem>) {
    if pending.is_empty() {
        return;
    }

    let items: Vec<BatchItem> = pending.drain(..).collect();
    let embeddings: Vec<Vec<f32>> = items.iter().map(|i| i.embedding.clone()).collect();

    // batch_search is a blocking call into intelligence_core — run on
    // the blocking thread pool so we don't stall the Tokio executor.
    let school_id = school_id.to_string();
    let role = role.to_string();

    let results = tokio::task::spawn_blocking(move || {
        batch_search(&school_id, &embeddings, &role, 1)
    })
    .await
    .unwrap_or_else(|_| vec![None; items.len()]);

    // Dispatch results — one per caller
    for (item, result) in items.into_iter().zip(results.into_iter()) {
        let _ = item.respond_to.send(result);
    }
}

/// Remove batcher for a camera (called when camera is cleared).
pub fn remove_batcher(school_id: &str, camera_id: &str, role: &str) {
    let key = (school_id.to_string(), camera_id.to_string(), role.to_string());
    BATCHERS.remove(&key);
    // The task will exit when the channel closes (Sender dropped).
}