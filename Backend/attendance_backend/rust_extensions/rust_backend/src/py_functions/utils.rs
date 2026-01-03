// src/py_functions/utils.rs

use crate::{models::{ort_model, tch_model}, preprocess::preprocess_image, utils::cosine_similarity};
use pyo3::{exceptions::PyValueError, prelude::*};

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

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_face, m)?)?;
    m.add_function(wrap_pyfunction!(detect_emotion, m)?)?;
    Ok(())
}