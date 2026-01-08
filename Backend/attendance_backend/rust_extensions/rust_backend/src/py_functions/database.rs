use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use crate::rust_only::database::logic::*;

#[pyfunction]
fn check_duplicate(
    embedding: Vec<f32>,
    role: String,
    threshold: f32,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let result = check_duplicate_rust(&embedding, &role, threshold);

        let dict = PyDict::new(py);
        dict.set_item("duplicate", result.duplicate)?;

        if result.duplicate {
            dict.set_item("matched_id", result.matched_id)?;
            dict.set_item("similarity", result.similarity)?;
            dict.set_item("name", result.name)?;
            dict.set_item("roll_no", result.roll_no)?;
        }

        Ok(dict.to_object(py))
    })
}

#[pyfunction]
fn get_face_info(id: usize) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if let Some((id, name, roll_no, role)) = get_face_info_rust(id) {
            let dict = PyDict::new(py);
            dict.set_item("id", id)?;
            dict.set_item("name", name)?;
            dict.set_item("roll_no", roll_no)?;
            dict.set_item("role", role)?;
            Ok(dict.to_object(py))
        } else {
            Err(PyValueError::new_err("Face ID not found"))
        }
    })
}

#[pyfunction]
fn count_students() -> PyResult<usize> {
    Ok(count_students_rust())
}

#[pyfunction]
fn count_teachers() -> PyResult<usize> {
    Ok(count_teachers_rust())
}

#[pyfunction]
fn save_database(path: String) -> PyResult<()> {
    save_database_rust(&path)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn load_database(path: String) -> PyResult<()> {
    load_database_rust(&path)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn total_registered() -> PyResult<usize> {
    Ok(total_registered_rust())
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
