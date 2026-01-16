use chrono::NaiveTime;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::rust_only::scheduler::logic::{
    schedule_classes_beam_rust, schedule_classes_rust, PyClassInput,
};

fn parse_classes(py: Python, py_classes: Vec<PyObject>) -> PyResult<Vec<PyClassInput>> {
    let mut classes = Vec::new();

    for cls in py_classes {
        let dict: &PyDict = cls.extract(py)?;

        let start: String = dict.get_item("start_time").unwrap().extract()?;
        let end: String = dict.get_item("end_time").unwrap().extract()?;

        let start_time = NaiveTime::parse_from_str(&start, "%H:%M")
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let end_time = NaiveTime::parse_from_str(&end, "%H:%M")
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        classes.push(PyClassInput {
            class_name: dict.get_item("class_name").unwrap().extract()?,
            section: dict.get_item("section").unwrap().extract()?,
            subject: dict.get_item("subject").unwrap().extract()?,
            start_time,
            end_time,
        });
    }

    Ok(classes)
}

#[pyfunction]
pub fn schedule_classes(py_classes: Vec<PyObject>) -> PyResult<PyObject> {
    let inputs = Python::with_gil(|py| parse_classes(py, py_classes))?;

    let results = Python::with_gil(|py| py.allow_threads(|| schedule_classes_rust(inputs)));

    Python::with_gil(|py| {
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
    })
}

#[pyfunction]
pub fn schedule_classes_beam(py_classes: Vec<PyObject>) -> PyResult<PyObject> {
    // 1️⃣ Parse Python → Rust (GIL REQUIRED)
    let inputs = Python::with_gil(|py| parse_classes(py, py_classes))?;

    // 2️⃣ Run heavy scheduler WITHOUT GIL
    let results = Python::with_gil(|py| py.allow_threads(|| schedule_classes_beam_rust(inputs)));

    // 3️⃣ Convert Rust → Python (GIL REQUIRED)
    Python::with_gil(|py| {
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
    })
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(schedule_classes, m)?)?;
    m.add_function(wrap_pyfunction!(schedule_classes_beam, m)?)?;
    Ok(())
}
