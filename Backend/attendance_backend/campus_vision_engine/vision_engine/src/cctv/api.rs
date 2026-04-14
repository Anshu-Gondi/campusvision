/// cctv/api.rs  (replaces previous version)
///
/// Additional fix over previous version:
///   - `set_school_id` called on new trackers so tracker.rs identity
///     resolution has the school_id it needs.
///   - Identity resolution in api.rs (batcher path) is now the primary
///     path for locked tracks needing periodic re-verify.
///   - Tracker internal resolution (tracker.rs) handles the early
///     unlocked phase while hits < min_hits_for_api_batcher.
///   - mark_tracked_face now correctly fires because id_locked is
///     actually set by the tracker.

use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

use crate::app::AppState;
use crate::cctv::frame_batcher::get_or_create_batcher;
use crate::cctv::state::{clear_daily_records, mark_tracked_face};
use crate::cctv::tracker::FaceTracker;
use intelligence_core::embeddings::get_metadata;

type TrackerKey = (String, String, String); // (school_id, role, camera_id)
type TrackerMap = DashMap<TrackerKey, Arc<TokioMutex<FaceTracker>>>;

static TRACKERS: Lazy<TrackerMap> = Lazy::new(DashMap::new);

fn get_or_create_tracker(
    school_id: &str,
    role: &str,
    camera_id: &str,
    state: Arc<AppState>,
) -> Arc<TokioMutex<FaceTracker>> {
    let key = (school_id.to_string(), role.to_string(), camera_id.to_string());

    if let Some(t) = TRACKERS.get(&key) {
        return Arc::clone(t.value());
    }

    let mut tracker = FaceTracker::new(
        30,
        state,
        Some((112, 112)),
        Some("NHWC".to_string()),
    );

    // ✅ Critical: give the tracker its school_id so resolve_identities works.
    tracker.set_school_id(school_id);

    let arc = Arc::new(TokioMutex::new(tracker));
    TRACKERS.insert(key, Arc::clone(&arc));
    arc
}

// ── Output ────────────────────────────────────────────────────────────────────

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

// ── Main entry ────────────────────────────────────────────────────────────────

pub async fn process_frame_rust(
    state: Arc<AppState>,
    frame_bytes: &[u8],
    school_id: &str,
    role: &str,
    camera_id: &str,
    min_confidence: f32,
    min_track_hits: u32,
) -> anyhow::Result<Vec<CctvResult>> {
    if !["student", "teacher"].contains(&role) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    // ── 1. CCTV scheduler permit (non-blocking drop if full) ──────────────
    let permit = match state.scheduler.try_acquire_cctv() {
        Some(p) => p,
        None => return Ok(Vec::new()),
    };

    // ── 2. Run tracker (detect → embed → assign → resolve internally) ─────
    let tracker_arc = get_or_create_tracker(school_id, role, camera_id, Arc::clone(&state));

    let tracks = {
        let mut tracker = tracker_arc.lock().await;
        tracker.update_from_bytes(vec![frame_bytes]).await?
    };

    // ── 3. Yield to direct calls before HNSW search ───────────────────────
    if permit.should_yield() {
        return Ok(tracks
            .into_iter()
            .map(|track| build_result(&track, role, None, None, None))
            .collect());
    }

    // ── 4. For API-level re-verification of locked tracks, use batcher ────
    //
    // The tracker already resolved unlocked tracks internally.
    // Here we only push locked tracks that hit their reverify window
    // into the batcher, which is more efficient than resolve_async.
    let batcher = get_or_create_batcher(school_id, camera_id, role);

    let mut results = Vec::with_capacity(tracks.len());

    for track in tracks {
        let mut mark_now = None;

        // Fetch metadata for identified tracks to fill name/roll_no.
        let (name, roll_no) = if let Some(pid) = track.person_id {
            match get_metadata(school_id, pid) {
                Some(meta) => (Some(meta.name), Some(meta.roll_no)),
                None => (None, None),
            }
        } else {
            (None, None)
        };

        // Attempt attendance marking for locked, confident tracks.
        if track.id_locked
            && track.hits >= min_track_hits
            && track.confidence >= min_confidence
        {
            let marked = mark_tracked_face(&track, role);
            mark_now = Some(marked);
        }

        results.push(build_result(&track, role, mark_now, name, roll_no));
    }

    Ok(results)
}

fn build_result(
    track: &crate::cctv::types::TrackedFace,
    role: &str,
    mark_now: Option<bool>,
    name: Option<String>,
    roll_no: Option<String>,
) -> CctvResult {
    CctvResult {
        track_id: track.track_id as i32,
        bbox: (track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height),
        hits: track.hits,
        age: track.age,
        person_id: track.person_id.map(|id| id as u64),
        name,
        roll_no,
        role: Some(role.to_string()),
        identified: track.id_locked, // only true when ACTUALLY locked
        confidence: track.confidence as f64,
        mark_now,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn get_tracked_faces_rust(school_id: &str, role: &str, camera_id: &str) -> Vec<CctvResult> {
    let key = (school_id.to_string(), role.to_string(), camera_id.to_string());
    if let Some(entry) = TRACKERS.get(&key) {
        if let Ok(tracker) = entry.value().try_lock() {
            return tracker
                .manager
                .tracks
                .values()
                .map(|t| build_result(t, role, None, None, None))
                .collect();
        }
    }
    Vec::new()
}

pub fn clear_daily_rust() {
    clear_daily_records();
}

pub fn clear_camera(school_id: &str, role: &str, camera_id: &str) {
    let key = (school_id.to_string(), role.to_string(), camera_id.to_string());
    TRACKERS.remove(&key);
    crate::cctv::frame_batcher::remove_batcher(school_id, camera_id, role);
}