// src/py_functions/cctv.rs

use crate::{cctv_state::*, cctv_tracker::FaceTracker, hnsw_helper};
use once_cell::sync::Lazy;
use crate::preprocess;
use opencv::{core::{Point2f, Rect, Size, Vector}, imgcodecs, prelude::*};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use std::sync::Mutex;

static STUDENT_TRACKER: Lazy<Mutex<FaceTracker>> = Lazy::new(|| Mutex::new(FaceTracker::new(30)));
static TEACHER_TRACKER: Lazy<Mutex<FaceTracker>> = Lazy::new(|| Mutex::new(FaceTracker::new(30)));

#[pyfunction]
fn cctv_process_frame(
    frame_bytes: Vec<u8>,
    role: String,
    min_confidence: f32,
    min_track_hits: u32,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if !["student", "teacher"].contains(&role.as_str()) {
            return Err(PyValueError::new_err("role must be 'student' or 'teacher'"));
        }

        macro_rules! cv {
            ($e:expr) => {
                $e.map_err(|e| PyValueError::new_err(format!("OpenCV error: {}", e)))?
            };
        }
        macro_rules! any {
            ($e:expr) => {
                $e.map_err(|e| PyValueError::new_err(format!("Processing error: {}", e)))?
            };
        }

        let mat = cv!(imgcodecs::imdecode(
            &Vector::from_slice(&frame_bytes),
            imgcodecs::IMREAD_COLOR
        ));
        if mat.empty() {
            return Ok(Vec::<PyObject>::new().to_object(py));
        }

        let faces = any!(preprocess::detect_faces(
            &mat,
            "models/face_detection_yunet_2023mar.onnx",
            Size::new(320, 320),
            0.5,
        ));

        let mut detections = Vec::new();

        for row in 0..faces.rows() {
            let score = cv!(faces.at_2d::<f32>(row, 14));
            if *score < 0.5 {
                continue;
            }

            let x = *cv!(faces.at_2d::<f32>(row, 0)) as i32;
            let y = *cv!(faces.at_2d::<f32>(row, 1)) as i32;
            let w = *cv!(faces.at_2d::<f32>(row, 2)) as i32;
            let h = *cv!(faces.at_2d::<f32>(row, 3)) as i32;
            let bbox = Rect::new(x, y, w, h);

            let mut landmarks = Vec::with_capacity(5);
            for &col in &[4, 6, 8, 10, 12] {
                let px = cv!(faces.at_2d::<f32>(row, col));
                let py = cv!(faces.at_2d::<f32>(row, col + 1));
                landmarks.push(Point2f::new(*px, *py));
            }

            let face_tensor = any!(preprocess::preprocess_from_mat_and_landmarks(
                &mat, bbox, &landmarks
            ));

            detections.push((bbox, landmarks, face_tensor));
        }

        let tracker_mutex = if role == "teacher" {
            &TEACHER_TRACKER
        } else {
            &STUDENT_TRACKER
        };
        let mut tracker = tracker_mutex.lock().unwrap();
        let tracks = tracker.update(detections);

        let mut results = Vec::new();

        for track in &tracks {
            let dict = PyDict::new(py);

            let mut identified = track.person_id.is_some();
            let mut confidence = 0.0f64;

            if track.person_id.is_none() && track.hits >= min_track_hits {
                let search = hnsw_helper::search_in_role(&track.embedding, &role, 1);
                if let Some((id, sim)) = search.first() {
                    if *sim >= min_confidence {
                        if let Some(meta) = hnsw_helper::get_metadata(*id) {
                            dict.set_item("person_id", id)?;
                            dict.set_item("name", meta.name.clone())?;
                            dict.set_item("roll_no", meta.roll_no.clone())?;
                            dict.set_item("role", meta.role.clone())?;
                            dict.set_item("confidence", *sim as f64)?;
                            confidence = *sim as f64;
                            identified = true;

                            let should_mark = !is_already_marked_today(&role, *id);
                            dict.set_item("mark_now", should_mark)?;
                            if should_mark {
                                mark_person_today(&role, *id);
                            }
                        }
                    }
                }
            } else if let Some(id) = track.person_id {
                if let Some(meta) = hnsw_helper::get_metadata(id) {
                    dict.set_item("person_id", id)?;
                    dict.set_item("name", meta.name.clone())?;
                    dict.set_item("roll_no", meta.roll_no.clone())?;
                    dict.set_item("role", meta.role.clone())?;
                    dict.set_item("confidence", 0.99)?;
                    identified = true;
                }
            }

            dict.set_item("track_id", track.track_id as i32)?;
            dict.set_item(
                "bbox",
                (
                    track.bbox.x,
                    track.bbox.y,
                    track.bbox.width,
                    track.bbox.height,
                ),
            )?;
            dict.set_item("hits", track.hits)?;
            dict.set_item("age", track.age)?;
            dict.set_item("identified", identified)?;
            dict.set_item("confidence", confidence)?;

            results.push(dict.to_object(py));
        }

        Ok(results.to_object(py))
    })
}

#[pyfunction]
fn cctv_get_tracked_faces(role: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let tracker_mutex = if role == "teacher" {
            &TEACHER_TRACKER
        } else {
            &STUDENT_TRACKER
        };
        let tracker = tracker_mutex.lock().unwrap();
        let tracks = tracker.tracks.values().cloned().collect::<Vec<_>>();

        let mut results = Vec::new();
        for track in tracks {
            let dict = PyDict::new(py);
            dict.set_item("track_id", track.track_id as i32)?;
            dict.set_item("hits", track.hits)?;
            dict.set_item("age", track.age)?;
            dict.set_item("person_id", track.person_id.map(|id| id as i32))?;
            dict.set_item(
                "bbox",
                (
                    track.bbox.x,
                    track.bbox.y,
                    track.bbox.width,
                    track.bbox.height,
                ),
            )?;
            results.push(dict.to_object(py));
        }
        Ok(results.to_object(py))
    })
}

#[pyfunction]
fn cctv_clear_daily() -> PyResult<()> {
    clear_daily_records();
    Ok(())
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cctv_process_frame, m)?)?;
    m.add_function(wrap_pyfunction!(cctv_get_tracked_faces, m)?)?;
    m.add_function(wrap_pyfunction!(cctv_clear_daily, m)?)?;
    Ok(())
}