use crate::rust_only::detection::logic::*;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

#[pyfunction]
fn detect_and_embed(
    image_bytes: Vec<u8>,
    model_path: Option<String>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let model_ref = model_path.as_deref();

        let result = detect_and_embed_rust(&image_bytes, model_ref)
            .map_err(|e| {
                PyValueError::new_err(format!(
                    "detect_and_embed failed: {}",
                    e
                ))
            })?;

        let dict = PyDict::new(py);
        dict.set_item("found", result.found)?;

        if result.found {
            if let Some((x, y, w, h)) = result.bbox {
                let bbox = PyDict::new(py);
                bbox.set_item("x", x)?;
                bbox.set_item("y", y)?;
                bbox.set_item("w", w)?;
                bbox.set_item("h", h)?;
                dict.set_item("bbox", bbox)?;
            }

            if let Some(ref emb) = result.embedding {
                if emb.is_empty() {
                    return Err(PyValueError::new_err("Empty embedding vector"));
                }
                dict.set_item("embedding", emb)?;
            }
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
    detect_and_add_person_rust(&image_bytes, name, person_id, roll_no, role)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_and_embed, m)?)?;
    m.add_function(wrap_pyfunction!(detect_and_add_person, m)?)?;
    Ok(())
}
