use crate::cctv::state::*;
use crate::cctv::tracker::FaceTracker;
use crate::app::AppState;

use opencv::core::Vector;
use opencv::imgcodecs;
use opencv::prelude::*;

use serde::Serialize;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::Arc;

static TRACKERS: Lazy<DashMap<(String, String), FaceTracker>> =
    Lazy::new(|| DashMap::new());

static GLOBAL_STATE: Lazy<Arc<AppState>> =
    Lazy::new(|| {
        tokio::runtime::Handle::current()
            .block_on(AppState::new())
            .into()
    });

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

pub async fn process_frame_rust(
    frame_bytes: &[u8],
    role: &str,
    camera_id: &str,
    min_confidence: f32,
    min_track_hits: u32,
) -> anyhow::Result<Vec<CctvResult>> {

    if !["student", "teacher"].contains(&role) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    let mat = imgcodecs::imdecode(
        &Vector::from_slice(frame_bytes),
        imgcodecs::IMREAD_COLOR,
    )?;

    if mat.empty() {
        return Ok(Vec::new());
    }

    let key = (role.to_string(), camera_id.to_string());

    // 🔥 Get or create tracker (NO async inside lock)
    let mut tracker = TRACKERS
        .entry(key.clone())
        .or_insert_with(|| {
            FaceTracker::new(
                30,
                GLOBAL_STATE.clone(),
                Some((112, 112)),
                Some("NHWC".to_string()),
            )
        });

    // IMPORTANT: clone tracker to avoid holding DashMap lock during await
    let mut tracker_owned = tracker.clone();
    drop(tracker);

    // 🔥 Async update
    let tracks = tracker_owned
        .update_from_bytes(vec![frame_bytes])
        .await?;

    // Write back updated tracker
    TRACKERS.insert(key, tracker_owned);

    let mut results = Vec::with_capacity(tracks.len());

    for track in tracks {

        let mut mark_now = None;

        if track.hits >= min_track_hits
            && track.confidence >= min_confidence
        {
            let marked = mark_tracked_face(&track, role);
            mark_now = Some(marked);
        }

        results.push(CctvResult {
            track_id: track.track_id as i32,
            bbox: (
                track.bbox.x,
                track.bbox.y,
                track.bbox.width,
                track.bbox.height,
            ),
            hits: track.hits,
            age: track.age,
            person_id: track.person_id.map(|id| id as u64),
            name: None,
            roll_no: None,
            role: Some(role.to_string()),
            identified: track.person_id.is_some(),
            confidence: track.confidence as f64,
            mark_now,
        });
    }

    Ok(results)
}


pub fn get_tracked_faces_rust(
    role: &str,
    camera_id: &str,
) -> Vec<CctvResult> {

    let key = (role.to_string(), camera_id.to_string());

    if let Some(tracker) = TRACKERS.get(&key) {
        tracker.tracks
            .values()
            .map(|t| CctvResult {
                track_id: t.track_id as i32,
                bbox: (
                    t.bbox.x,
                    t.bbox.y,
                    t.bbox.width,
                    t.bbox.height,
                ),
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
    } else {
        Vec::new()
    }
}

pub fn clear_daily_rust() {
    clear_daily_records();
}

pub fn clear_camera(role: &str, camera_id: &str) {
    TRACKERS.remove(&(role.to_string(), camera_id.to_string()));
}
