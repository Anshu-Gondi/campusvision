// src/py_functions/utils.rs

use pyo3::{exceptions::PyValueError, prelude::*};
use crate::rust_only::utils::logic::*;

#[pyfunction]
fn verify_face(
    input_image: Vec<u8>,
    known_embedding: Vec<f32>,
) -> PyResult<f32> {
    verify_face_rust(input_image, known_embedding)
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn detect_emotion(
    input_image: Vec<u8>,
) -> PyResult<i64> {
    detect_emotion_rust(input_image)
        .map_err(|e| PyValueError::new_err(e))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_face, m)?)?;
    m.add_function(wrap_pyfunction!(detect_emotion, m)?)?;
    Ok(())
}
