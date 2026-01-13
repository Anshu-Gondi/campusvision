use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use crate::rust_only::cctv::logic::{process_frame_rust, get_tracked_faces_rust, clear_daily_rust};

#[pyfunction]
fn cctv_process_frame(
    frame_bytes: Vec<u8>,
    role: String,
    min_confidence: f32,
    min_track_hits: u32,
    model_path: Option<String>, // optional
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let model_ref = model_path.as_deref(); // Option<String> -> Option<&str>
        let results = process_frame_rust(&frame_bytes, &role, min_confidence, min_track_hits, model_ref)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        let py_list = pyo3::types::PyList::empty(py);
        for r in results {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("track_id", r.track_id)?;
            dict.set_item("bbox", r.bbox)?;
            dict.set_item("hits", r.hits)?;
            dict.set_item("age", r.age)?;
            dict.set_item("person_id", r.person_id)?;
            dict.set_item("name", r.name)?;
            dict.set_item("roll_no", r.roll_no)?;
            dict.set_item("role", r.role)?;
            dict.set_item("identified", r.identified)?;
            dict.set_item("confidence", r.confidence)?;
            dict.set_item("mark_now", r.mark_now)?;
            py_list.append(dict)?;
        }

        Ok(py_list.to_object(py))
    })
}

#[pyfunction]
fn cctv_get_tracked_faces(role: String) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let results = get_tracked_faces_rust(&role);
        let py_list = PyList::empty(py);
        for r in results {
            let dict = PyDict::new(py);
            dict.set_item("track_id", r.track_id)?;
            dict.set_item("bbox", r.bbox)?;
            dict.set_item("hits", r.hits)?;
            dict.set_item("age", r.age)?;
            dict.set_item("person_id", r.person_id)?;
            dict.set_item("identified", r.identified)?;
            dict.set_item("confidence", r.confidence)?;
            py_list.append(dict)?;
        }
        Ok(py_list.to_object(py))
    })
}

#[pyfunction]
fn cctv_clear_daily() -> PyResult<()> {
    clear_daily_rust();
    Ok(())
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cctv_process_frame, m)?)?;
    m.add_function(wrap_pyfunction!(cctv_get_tracked_faces, m)?)?;
    m.add_function(wrap_pyfunction!(cctv_clear_daily, m)?)?;
    Ok(())
}
