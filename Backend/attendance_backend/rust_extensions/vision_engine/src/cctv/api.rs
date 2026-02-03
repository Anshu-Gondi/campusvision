// src/cctv/api.rs

// src/cctv/api.rs

use crate::cctv::state::*;
use crate::cctv::tracker::{FaceTracker, TrackedFace}; // <- import the trait for update
use crate::preprocessing;
use intelligence_core::embeddings::{batch_search, get_metadata};
use opencv::core::{Size, Vector};
use opencv::imgcodecs;
use opencv::prelude::MatTraitConst;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// (role, camera_id) -> FaceTracker
    static TRACKERS: RefCell<HashMap<(String, String), FaceTracker>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone, Serialize)]
pub struct CctvResult {
    pub track_id: i32,
    pub bbox: (i32, i32, i32, i32),
    pub hits: u32,
    pub age: u32,
    pub person_id: Option<u64>,
    pub name: Option<String>,
    pub roll_no: Option<String>,
    pub role: Option<String>,
    pub identified: bool,
    pub confidence: f64,
    pub mark_now: Option<bool>,
}

pub fn process_frame_rust(
    frame_bytes: &[u8],               // ← raw BGR bytes from client/camera
    role: &str,
    camera_id: &str,
    min_confidence: f32,
    min_track_hits: u32,
    model_path: Option<&str>,
) -> anyhow::Result<Vec<CctvResult>> {
    if !["student", "teacher"].contains(&role) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    // Decode raw bytes → Mat (one frame for simplicity; extend for multi-frame later)
    let mat = imgcodecs::imdecode(&Vector::from_slice(frame_bytes), imgcodecs::IMREAD_COLOR)?;
    if mat.empty() {
        return Ok(Vec::new());
    }

    // Pass SINGLE raw frame as Vec<&[u8]>
    let raw_frames = vec![frame_bytes];

    let tracks = TRACKERS.with(|map| {
        let mut map = map.borrow_mut();
        let key = (role.to_string(), camera_id.to_string());
        let tracker = map.entry(key).or_insert_with(|| FaceTracker::new(30));

        // FIXED: Pass raw bytes, NOT detections
        tracker.update_from_bytes(raw_frames)
    });

    // Convert tracks to your CctvResult format (your existing logic)
    let mut results = Vec::with_capacity(tracks.len());

    for track in tracks {
        let mut result = CctvResult {
            track_id: track.track_id as i32,
            bbox: (track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height),
            hits: track.hits,
            age: track.age,
            person_id: track.person_id.map(|id| id as u64),
            name: None,
            roll_no: None,
            role: Some(role.to_string()),
            identified: track.person_id.is_some(),
            confidence: track.confidence as f64,
            mark_now: None,
        };

        // Batch identification logic (your existing code)
        // ... paste your candidate_embeddings / batch_results / metadata lookup here ...

        results.push(result);
    }

    Ok(results)
}

pub fn get_tracked_faces_rust(role: &str, camera_id: &str) -> Vec<CctvResult> {
    TRACKERS.with(|map| {
        map.borrow()
            .get(&(role.to_string(), camera_id.to_string()))
            .map(|tracker| {
                tracker
                    .tracks
                    .values()
                    .map(|t| CctvResult {
                        track_id: t.track_id as i32,
                        bbox: (t.bbox.x, t.bbox.y, t.bbox.width, t.bbox.height),
                        hits: t.hits,
                        age: t.age,
                        person_id: t.person_id.map(|id| id as u64),
                        name: None,
                        roll_no: None,
                        role: None,
                        identified: t.person_id.is_some(),
                        confidence: t.confidence as f64,
                        mark_now: None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

pub fn clear_daily_rust() {
    clear_daily_records();
}

pub fn clear_camera(role: &str, camera_id: &str) {
    TRACKERS.with(|map| {
        map.borrow_mut()
            .remove(&(role.to_string(), camera_id.to_string()));
    });
}