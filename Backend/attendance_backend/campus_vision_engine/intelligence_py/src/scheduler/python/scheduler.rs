use chrono::NaiveTime;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::scheduler::api::{schedule_classes_beam_rust, schedule_classes_rust, PyClassInput};

/// Helper: get a required key from Python dict
fn get_required<'py, T: FromPyObject<'py>>(dict: &'py PyDict, key: &str) -> PyResult<T> {
    dict.get_item(key)
        .ok_or_else(|| PyValueError::new_err(format!("Missing key: {key}")))?
        .extract()
}

/// Parse Python list of class dicts into Rust Vec<PyClassInput>
fn parse_classes(py: Python, py_classes: Vec<PyObject>) -> PyResult<Vec<PyClassInput>> {
    let mut classes = Vec::with_capacity(py_classes.len());

    for cls in py_classes {
        let dict: &PyDict = cls.extract(py)?;

        let start: String = get_required(dict, "start_time")?;
        let end: String = get_required(dict, "end_time")?;

        let start_time = NaiveTime::parse_from_str(&start, "%H:%M")
            .map_err(|e| PyValueError::new_err(format!("Invalid start_time: {e}")))?;
        let end_time = NaiveTime::parse_from_str(&end, "%H:%M")
            .map_err(|e| PyValueError::new_err(format!("Invalid end_time: {e}")))?;

        classes.push(PyClassInput {
            class_name: get_required(dict, "class_name")?,
            section: get_required(dict, "section")?,
            subject: get_required(dict, "subject")?,
            start_time,
            end_time,
        });
    }

    Ok(classes)
}

#[pyfunction]
pub fn schedule_classes(py: Python, py_classes: Vec<PyObject>) -> PyResult<PyObject> {
    let inputs = parse_classes(py, py_classes)?;

    let results = py.allow_threads(|| schedule_classes_rust(inputs));

    let py_list = pyo3::types::PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("class_name", r.class_name)?;
        d.set_item("section", r.section)?;
        d.set_item("subject", r.subject)?;
        d.set_item("teacher_id", r.teacher_id)?;
        d.set_item("teacher_name", r.teacher_name)?;
        d.set_item("similarity", r.similarity)?;
        d.set_item("reliability", r.reliability)?;
        d.set_item("workload", r.workload)?;
        py_list.append(d)?;
    }

    Ok(py_list.to_object(py))
}

#[pyfunction]
pub fn schedule_classes_beam(py: Python, py_classes: Vec<PyObject>) -> PyResult<PyObject> {
    let inputs = parse_classes(py, py_classes)?;

    let results = py.allow_threads(|| schedule_classes_beam_rust(inputs));

    let py_list = pyo3::types::PyList::empty(py);
    for r in results {
        let d = PyDict::new(py);
        d.set_item("class_name", r.class_name)?;
        d.set_item("section", r.section)?;
        d.set_item("subject", r.subject)?;
        d.set_item("teacher_id", r.teacher_id)?;
        d.set_item("teacher_name", r.teacher_name)?;
        d.set_item("similarity", r.similarity)?;
        d.set_item("reliability", r.reliability)?;
        d.set_item("workload", r.workload)?;
        py_list.append(d)?;
    }

    Ok(py_list.to_object(py))
}

/// Make this function public so lib.rs can call it
pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(schedule_classes, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_classes_beam, m)?)?;
    Ok(())
}
