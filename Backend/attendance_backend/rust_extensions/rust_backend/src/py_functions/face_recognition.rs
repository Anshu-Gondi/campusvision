// src/py_functions/face_recognition.rs

use crate::hnsw_helper;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
fn add_person(embedding: Vec<f32>, name: String, person_id: u64, roll_no: String, role: String) -> PyResult<usize> {
    if !["student", "teacher"].contains(&role.as_str()) {
        return Err(PyValueError::new_err("role must be 'student' or 'teacher'"));
    }
    hnsw_helper::add_face_embedding(embedding, name, person_id, roll_no, role)
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
fn can_reenroll(
    embedding: Vec<f32>,
    person_id: u64,
    role: String,
) -> PyResult<bool> {
    hnsw_helper::can_reenroll(&embedding, person_id, &role)
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
    let id = hnsw_helper::add_face_embedding(
        embedding, name, person_id, // ← FIXED
        roll_no, role,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(id)
}

#[pyfunction]
fn query_similar(embedding: Vec<f32>, k: usize) -> PyResult<Vec<usize>> {
    let results = hnsw_helper::search_in_role(&embedding, "student", k);
    Ok(results.into_iter().map(|(id, _)| id).collect())
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_person, m)?)?;
    m.add_function(wrap_pyfunction!(search_person, m)?)?;
    m.add_function(wrap_pyfunction!(can_reenroll, m)?)?;
    m.add_function(wrap_pyfunction!(add_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(query_similar, m)?)?;
    Ok(())
}