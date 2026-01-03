// src/py_functions/detection.rs

use crate::{models::{ort_model, tch_model}, preprocess, hnsw_helper};
use opencv::{core::{Vector, Size}, imgcodecs};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

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

#[pyfunction]
fn detect_and_add_person(
    image_bytes: Vec<u8>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String,
) -> PyResult<usize> {
    if !["student", "teacher"].contains(&role.as_str()) {
        return Err(PyValueError::new_err("role must be 'student' or 'teacher'"));
    }

    // 1. Detect + embed
    let result = detect_and_embed(image_bytes)?;

    Python::with_gil(|py| {
        let dict: &PyDict = result.extract(py)?;

        let found: bool = dict.get_item("found").unwrap().extract()?;
        if !found {
            return Err(PyValueError::new_err("No face detected"));
        }

        let embedding: Vec<f32> = dict.get_item("embedding").unwrap().extract()?;

        // 2. Add to HNSW
        let id = hnsw_helper::add_face_embedding(embedding, name, person_id, roll_no, role)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(id)
    })
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_and_embed, m)?)?;
    m.add_function(wrap_pyfunction!(detect_and_add_person, m)?)?;
    Ok(())
}