use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use crate::rust_only::detection::logic::*;

#[pyfunction]
fn detect_and_embed(
    image_bytes: Vec<u8>,
    model_path: Option<String>, // optional
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let model_ref = model_path.as_deref(); // convert Option<String> -> Option<&str>
        let result = detect_and_embed_rust(&image_bytes, model_ref)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("found", result.found)?;

        if result.found {
            dict.set_item("bbox", result.bbox)?;
            dict.set_item("embedding", result.embedding)?;
        }

        Ok(dict.to_object(py))
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
    detect_and_add_person_rust(
        &image_bytes,
        name,
        person_id,
        roll_no,
        role,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_and_embed, m)?)?;
    m.add_function(wrap_pyfunction!(detect_and_add_person, m)?)?;
    Ok(())
}
