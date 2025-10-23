#![allow(unsafe_op_in_unsafe_fn)]

mod preprocess;
mod models;
mod hnsw_helper;
mod utils;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use models::{tch_model, ort_model};
use preprocess::preprocess_image;
use utils::cosine_similarity;

#[pyfunction]
fn verify_face(input_image: Vec<u8>, known_embedding: Vec<f32>) -> PyResult<f32> {
    let img_tensor = preprocess_image(&input_image)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // 🔁 Either use Tch or ONNXRuntime backend:
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
    let img_tensor = preprocess_image(&input_image)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    // Try Tch backend first, fallback to ONNX if fails
    let result = if let Ok(r) = tch_model::run_emotion_model(&img_tensor) {
        r
    } else {
        ort_model::run_emotion_model_onnx(&img_tensor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
    };

    Ok(result)
}

#[pyfunction]
fn add_to_index(embedding: Vec<f32>) -> PyResult<()> {
    hnsw_helper::add_embedding(&embedding)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn query_similar(embedding: Vec<f32>, k: usize) -> PyResult<Vec<usize>> {
    Ok(hnsw_helper::query_embedding(&embedding, k))
}

#[pymodule]
fn rust_backend(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_face, m)?)?;
    m.add_function(wrap_pyfunction!(detect_emotion, m)?)?;
    m.add_function(wrap_pyfunction!(add_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(query_similar, m)?)?;
    Ok(())
}
