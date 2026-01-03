// src/py_functions/database.rs

use crate::hnsw_helper;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyfunction]
fn check_duplicate(
    embedding: Vec<f32>,
    role: String,
    threshold: f32,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let results = hnsw_helper::search_in_role(&embedding, &role, 1);

        let dict = PyDict::new(py);

        if let Some((id, sim)) = results.first() {
            if *sim >= threshold {
                if let Some(meta) = hnsw_helper::get_metadata(*id) {
                    dict.set_item("duplicate", true)?;
                    dict.set_item("matched_id", id)?;
                    dict.set_item("similarity", *sim)?;
                    dict.set_item("name", meta.name)?;
                    dict.set_item("roll_no", meta.roll_no)?;
                    return Ok(dict.into());
                }
            }
        }

        dict.set_item("duplicate", false)?;
        Ok(dict.into())
    })
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

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_duplicate, m)?)?;
    m.add_function(wrap_pyfunction!(get_face_info, m)?)?;
    m.add_function(wrap_pyfunction!(count_students, m)?)?;
    m.add_function(wrap_pyfunction!(count_teachers, m)?)?;
    m.add_function(wrap_pyfunction!(save_database, m)?)?;
    m.add_function(wrap_pyfunction!(load_database, m)?)?;
    m.add_function(wrap_pyfunction!(total_registered, m)?)?;
    Ok(())
}