#![allow(unsafe_op_in_unsafe_fn)]

mod cctv_state;
mod cctv_tracker;
mod hnsw_helper;
mod models;
mod preprocess;
mod scheduler;
mod utils;

use crate::scheduler::scheduler::{ClassRequest, FullScheduler, GraphScheduler};
use cctv_state::*;
use cctv_tracker::FaceTracker;
use chrono::NaiveTime;
use models::{ort_model, tch_model};
use once_cell::sync::Lazy;
use preprocess::preprocess_image;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Mutex;
use utils::cosine_similarity;

// ── ADD THESE IMPORTS (CRITICAL FIXES) ─────────────────────────────────────
use opencv::{
    core::{Point2f, Rect, Size, Vector},
    imgcodecs,
    prelude::*,
};

// One tracker per role (you can make per-camera later if needed)
static STUDENT_TRACKER: Lazy<Mutex<FaceTracker>> = Lazy::new(|| Mutex::new(FaceTracker::new(30)));
static TEACHER_TRACKER: Lazy<Mutex<FaceTracker>> = Lazy::new(|| Mutex::new(FaceTracker::new(30)));

// ──────────────────────────────────────────────────────────────
// Existing functions (unchanged)
// ──────────────────────────────────────────────────────────────

#[pyfunction]
fn verify_face(input_image: Vec<u8>, known_embedding: Vec<f32>) -> PyResult<f32> {
    let img_tensor =
        preprocess_image(&input_image).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let emb_vec = if let Ok(v) = tch_model::run_face_model(&img_tensor) {
        v
    } else {
        ort_model::run_face_model_onnx(&img_tensor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    Ok(cosine_similarity(&emb_vec, &known_embedding))
}

#[pyfunction]
fn detect_emotion(input_image: Vec<u8>) -> PyResult<i64> {
    let img_tensor =
        preprocess_image(&input_image).map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result = if let Ok(r) = tch_model::run_emotion_model(&img_tensor) {
        r
    } else {
        ort_model::run_emotion_model_onnx(&img_tensor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    Ok(result)
}

// ──────────────────────────────────────────────────────────────
// NEW: Teacher + Student functions (unchanged except one fix below)
// ──────────────────────────────────────────────────────────────

#[pyfunction]
fn add_person(embedding: Vec<f32>, name: String, roll_no: String, role: String) -> PyResult<usize> {
    if !["student", "teacher"].contains(&role.as_str()) {
        return Err(PyValueError::new_err("role must be 'student' or 'teacher'"));
    }
    hnsw_helper::add_embedding(embedding, name, roll_no, role)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn search_person(embedding: Vec<f32>, role: String, k: usize) -> PyResult<Vec<(usize, f32)>> {
    if !["student", "teacher"].contains(&role.as_str()) {
        return Err(PyValueError::new_err("role must be 'student' or 'teacher'"));
    }
    Ok(hnsw_helper::search_in_role(&embedding, &role, k))
}

#[pyfunction]
fn get_face_info(id: usize) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if let Some(meta) = hnsw_helper::get_metadata(id) {
            let dict = PyDict::new(py);
            dict.set_item("id", meta.id)?;
            dict.set_item("name", meta.name)?;
            dict.set_item("roll_no", meta.roll_no)?;
            dict.set_item("role", meta.role)?;
            Ok(dict.into())
        } else {
            Err(PyValueError::new_err("Face ID not found"))
        }
    })
}

#[pyfunction]
fn count_students() -> PyResult<usize> {
    Ok(hnsw_helper::count_by_role("student"))
}

#[pyfunction]
fn count_teachers() -> PyResult<usize> {
    Ok(hnsw_helper::count_by_role("teacher"))
}

#[pyfunction]
fn save_database(path: String) -> PyResult<()> {
    hnsw_helper::save_all(&path).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn load_database(path: String) -> PyResult<()> {
    hnsw_helper::load_all(&path).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn total_registered() -> PyResult<usize> {
    Ok(hnsw_helper::get_total_faces())
}

// ── FIXED detect_and_embed (uses correct OpenCV + error handling) ────────
#[pyfunction]
fn detect_and_embed(image_bytes: Vec<u8>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Decode once
        let mat = imgcodecs::imdecode(&Vector::from_slice(&image_bytes), imgcodecs::IMREAD_COLOR)
            .map_err(|e| PyValueError::new_err(format!("OpenCV decode failed: {}", e)))?;

        // Detect faces using YuNet (returns matrix of detections)
        let faces = preprocess::detect_faces(
            &mat,
            "models/face_detection_yunet_2023mar.onnx",
            Size::new(320, 320),
            0.6,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let best =
            preprocess::pick_best_face(&faces).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);

        if let Some((rect, landmarks)) = best {
            // preprocess using the already-detected bbox & landmarks (avoid double detection)
            let tensor = preprocess::preprocess_from_mat_and_landmarks(&mat, rect, &landmarks)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;

            // Prefer torchscript model (tch_model -> TorchScript .pt); fall back to ONNX if needed
            let embedding = match tch_model::run_face_model(&tensor) {
                Ok(v) => v,
                Err(_) => ort_model::run_face_model_onnx(&tensor)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            };

            dict.set_item("found", true)?;
            dict.set_item("bbox", (rect.x, rect.y, rect.width, rect.height))?;
            dict.set_item("embedding", embedding)?;
        } else {
            dict.set_item("found", false)?;
        }

        Ok(dict.into())
    })
}

// ── FIXED add_to_index (was returning wrong type before) ─────────────────
#[pyfunction]
fn add_to_index(embedding: Vec<f32>) -> PyResult<()> {
    hnsw_helper::add_embedding(
        embedding,
        "Unknown".to_string(),
        "NA".to_string(),
        "student".to_string(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
fn query_similar(embedding: Vec<f32>, k: usize) -> PyResult<Vec<usize>> {
    let results = hnsw_helper::search_in_role(&embedding, "student", k);
    Ok(results.into_iter().map(|(id, _)| id).collect())
}

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

            let tensor = any!(preprocess::preprocess_from_mat_and_landmarks(
                &mat, bbox, &landmarks
            ));
            let embedding = match tch_model::run_face_model(&tensor) {
                Ok(v) => v,
                Err(_) => ort_model::run_face_model_onnx(&tensor)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            };

            detections.push((bbox, landmarks, embedding));
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

#[pyfunction]
pub fn schedule_classes(py_classes: Vec<PyObject>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Convert Python → Rust structs
        let mut classes = Vec::new();

        for cls in py_classes {
            let dict: &pyo3::types::PyDict = cls.extract(py)?;

            let class_name: String = dict.get_item("class_name").unwrap().extract()?;
            let section: String = dict.get_item("section").unwrap().extract()?;
            let subject: String = dict.get_item("subject").unwrap().extract()?;

            let start: String = dict.get_item("start_time").unwrap().extract()?;
            let end: String = dict.get_item("end_time").unwrap().extract()?;

            let start_time = NaiveTime::parse_from_str(&start, "%H:%M")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

            let end_time = NaiveTime::parse_from_str(&end, "%H:%M")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

            classes.push(ClassRequest {
                class_name,
                section,
                subject,
                start_time,
                end_time,
            });
        }

        // --------------------------
        // Use your updated scheduler
        // --------------------------
        //
        // Choose embedding dimension (32 or 64 recommended)
        //
        let embedding_dim = 32;

        let mut scheduler = FullScheduler::new(embedding_dim);

        // Compute schedule
        let results = scheduler.assign_classes(&classes);

        // Convert Rust output → Python list of dicts
        let py_list = pyo3::types::PyList::empty(py);

        for r in results {
            let d = pyo3::types::PyDict::new(py);

            d.set_item("class_name", r.class.class_name)?;
            d.set_item("section", r.class.section)?;
            d.set_item("subject", r.class.subject)?;
            d.set_item("teacher_id", r.teacher.id)?;
            d.set_item("teacher_name", r.teacher.name)?;
            d.set_item("similarity", r.teacher.similarity)?;
            d.set_item("reliability", r.teacher.reliability)?;
            d.set_item("workload", r.teacher.workload)?;

            py_list.append(d)?;
        }

        Ok(py_list.to_object(py))
    })
}

#[pyfunction]
pub fn schedule_classes_beam(py_classes: Vec<PyObject>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // -------------------------
        // Convert Python → Rust
        // -------------------------
        let mut classes = Vec::new();

        for cls in py_classes {
            let dict: &pyo3::types::PyDict = cls.extract(py)?;

            let class_name: String = dict.get_item("class_name").unwrap().extract()?;
            let section: String = dict.get_item("section").unwrap().extract()?;
            let subject: String = dict.get_item("subject").unwrap().extract()?;

            let start: String = dict.get_item("start_time").unwrap().extract()?;
            let end: String = dict.get_item("end_time").unwrap().extract()?;

            let start_time = NaiveTime::parse_from_str(&start, "%H:%M")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

            let end_time = NaiveTime::parse_from_str(&end, "%H:%M")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

            classes.push(ClassRequest {
                class_name,
                section,
                subject,
                start_time,
                end_time,
            });
        }

        // --------------------------------------
        // Initialise your Beam Search Scheduler
        // --------------------------------------
        let embedding_dim = 32;

        let mut scheduler = GraphScheduler::new(
            embedding_dim,
            60,     // beam width (increase for higher accuracy)
            0.02,   // similarity threshold
        );

        // Run beam scheduler
        let results = scheduler.assign_classes_beam(&classes);

        // -------------------------
        // Convert Rust → Python
        // -------------------------
        let py_list = pyo3::types::PyList::empty(py);

        for r in results {
            let d = pyo3::types::PyDict::new(py);

            d.set_item("class_name", r.class.class_name)?;
            d.set_item("section", r.class.section)?;
            d.set_item("subject", r.class.subject)?;

            d.set_item("teacher_id", r.teacher.id)?;
            d.set_item("teacher_name", r.teacher.name)?;

            d.set_item("similarity", r.teacher.similarity)?;
            d.set_item("reliability", r.teacher.reliability)?;
            d.set_item("workload", r.teacher.workload)?;

            py_list.append(d)?;
        }

        Ok(py_list.to_object(py))
    })
}

// ──────────────────────────────────────────────────────────────
// Module registration
// ──────────────────────────────────────────────────────────────

#[pymodule]
fn rust_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_face, m)?)?;
    m.add_function(wrap_pyfunction!(detect_emotion, m)?)?;

    m.add_function(wrap_pyfunction!(add_person, m)?)?;
    m.add_function(wrap_pyfunction!(search_person, m)?)?;
    m.add_function(wrap_pyfunction!(get_face_info, m)?)?;
    m.add_function(wrap_pyfunction!(save_database, m)?)?;
    m.add_function(wrap_pyfunction!(load_database, m)?)?;
    m.add_function(wrap_pyfunction!(total_registered, m)?)?;
    m.add_function(wrap_pyfunction!(detect_and_embed, m)?)?;
    m.add_function(wrap_pyfunction!(count_students, m)?)?;
    m.add_function(wrap_pyfunction!(count_teachers, m)?)?;
    m.add_function(wrap_pyfunction!(cctv_process_frame, m)?)?;
    m.add_function(wrap_pyfunction!(cctv_clear_daily, m)?)?;
    m.add_function(wrap_pyfunction!(cctv_get_tracked_faces, m)?)?;

    m.add_function(wrap_pyfunction!(schedule_classes, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_classes_beam, m)?)?;

    m.add_function(wrap_pyfunction!(add_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(query_similar, m)?)?;

    Ok(())
}
