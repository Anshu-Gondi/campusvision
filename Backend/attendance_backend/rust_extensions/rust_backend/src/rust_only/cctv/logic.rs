use crate::cctv_state::*;
use crate::cctv_tracker::FaceTracker;
use crate::hnsw_helper;
use crate::preprocess;
use opencv::core::{Size, Vector};
use opencv::imgcodecs;
use opencv::prelude::MatTraitConst;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// (role, camera_id) -> FaceTracker
    static TRACKERS: RefCell<HashMap<(String, String), FaceTracker>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone)]
pub struct CctvResult {
    pub track_id: i32,
    pub bbox: (i32, i32, i32, i32),
    pub hits: u32,
    pub age: u32,
    pub person_id: Option<u64>, // public-facing
    pub name: Option<String>,
    pub roll_no: Option<String>,
    pub role: Option<String>,
    pub identified: bool,
    pub confidence: f64,
    pub mark_now: Option<bool>,
}

pub fn process_frame_rust(
    frame_bytes: &[u8],
    role: &str,
    camera_id: &str,
    min_confidence: f32,
    min_track_hits: u32,
    model_path: Option<&str>, // NEW optional argument
) -> anyhow::Result<Vec<CctvResult>> {
    if !["student", "teacher"].contains(&role) {
        anyhow::bail!("role must be 'student' or 'teacher'");
    }

    let mat = imgcodecs::imdecode(&Vector::from_slice(frame_bytes), imgcodecs::IMREAD_COLOR)?;
    if mat.empty() {
        return Ok(Vec::new());
    }

    let faces = preprocess::detect_faces(&mat, model_path, Size::new(320, 320), 0.5)?;

    let mut detections = Vec::new();
    for row in 0..faces.rows() {
        let score = *faces.at_2d::<f32>(row, 14)?;
        if score < 0.5 {
            continue;
        }

        let bbox = opencv::core::Rect::new(
            *faces.at_2d::<f32>(row, 0)? as i32,
            *faces.at_2d::<f32>(row, 1)? as i32,
            *faces.at_2d::<f32>(row, 2)? as i32,
            *faces.at_2d::<f32>(row, 3)? as i32,
        );

        let mut landmarks = Vec::with_capacity(5);
        for &col in &[4, 6, 8, 10, 12] {
            landmarks.push(opencv::core::Point2f::new(
                *faces.at_2d::<f32>(row, col)?,
                *faces.at_2d::<f32>(row, col + 1)?,
            ));
        }

        let tensor = preprocess::preprocess_from_mat_and_landmarks(&mat, bbox, &landmarks)?;
        detections.push((bbox, landmarks, tensor));
    }

    let tracks = TRACKERS.with(|map| {
        let mut map = map.borrow_mut();

        let key = (role.to_string(), camera_id.to_string());

        let tracker = map.entry(key).or_insert_with(|| FaceTracker::new(30));

        tracker.update(detections)
    });

    // build CctvResult as before
    let mut results = Vec::new();
    for track in tracks {
        let mut result = CctvResult {
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
            confidence: 0.0,
            mark_now: None,
        };

        // 🔍 Identification logic remains the same
        if track.person_id.is_none() && track.hits >= min_track_hits {
            if let Some((internal_id, sim)) =
                hnsw_helper::search_in_role(&track.embedding, role, 1).first()
            {
                if *sim >= min_confidence {
                    if let Some(meta) = hnsw_helper::get_metadata(*internal_id) {
                        let public_id = *internal_id as u64;
                        result.person_id = Some(public_id);
                        result.name = Some(meta.name.clone());
                        result.roll_no = Some(meta.roll_no.clone());
                        result.role = Some(meta.role.clone());
                        result.confidence = *sim as f64;
                        result.identified = true;

                        let should_mark = !is_already_marked_today(role, *internal_id);
                        result.mark_now = Some(should_mark);

                        if should_mark {
                            mark_person_today(role, *internal_id);
                        }
                    }
                }
            }
        }
        // Already identified tracks logic
        else if let Some(internal_id) = track.person_id {
            if let Some(meta) = hnsw_helper::get_metadata(internal_id) {
                result.person_id = Some(internal_id as u64);
                result.name = Some(meta.name.clone());
                result.roll_no = Some(meta.roll_no.clone());
                result.role = Some(meta.role.clone());
                result.confidence = 0.99;
            }
        }

        results.push(result);
    }

    Ok(results)
}

pub fn get_tracked_faces_rust(role: &str, camera_id: &str) -> Vec<CctvResult> {
    TRACKERS.with(|map| {
        let map = map.borrow();
        if let Some(tracker) = map.get(&(role.to_string(), camera_id.to_string())) {
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
        } else {
            Vec::new()
        }
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
