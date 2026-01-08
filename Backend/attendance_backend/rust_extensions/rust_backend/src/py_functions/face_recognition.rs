// src/py_functions/face_recognition.rs

use pyo3::{exceptions::PyValueError, prelude::*};
use crate::rust_only::face_recognition::logic::*;

#[pyfunction]
fn add_person(
    embedding: Vec<f32>,
    name: String,
    person_id: u64,
    roll_no: String,
    role: String,
) -> PyResult<usize> {
    add_person_rust(embedding, name, person_id, roll_no, role)
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn search_person(
    embedding: Vec<f32>,
    role: String,
    k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    search_person_rust(embedding, role, k)
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn can_reenroll(
    embedding: Vec<f32>,
    person_id: u64,
    role: String,
) -> PyResult<bool> {
    can_reenroll_rust(embedding, person_id, role)
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn add_to_index(
    embedding: Vec<f32>,
    person_id: u64,
    name: String,
    roll_no: String,
    role: String,
) -> PyResult<usize> {
    add_to_index_rust(embedding, person_id, name, roll_no, role)
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn query_similar(
    embedding: Vec<f32>,
    k: usize,
) -> PyResult<Vec<usize>> {
    Ok(query_similar_rust(embedding, k))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_person, m)?)?;
    m.add_function(wrap_pyfunction!(search_person, m)?)?;
    m.add_function(wrap_pyfunction!(can_reenroll, m)?)?;
    m.add_function(wrap_pyfunction!(add_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(query_similar, m)?)?;
    Ok(())
}
