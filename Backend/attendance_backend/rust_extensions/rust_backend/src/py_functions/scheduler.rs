// src/py_functions/scheduler.rs

use crate::scheduler::scheduler::{ClassRequest, FullScheduler, GraphScheduler};
use chrono::NaiveTime;
use pyo3::prelude::*;

#[pyfunction]
pub fn schedule_classes(py_classes: Vec<PyObject>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // Convert Python → Rust structs
        let mut classes = Vec::new();

        for cls in py_classes {
            let dict: &pyo3::types::PyDict = cls.extract(py)?;

            let class_name: String = dict.get_item("class_name").unwrap().extract()?;
            let section: String = dict.get_item("section").unwrap().extract()?;
            let subject: String = dict.get_item("subject").unwrap().extract()?;

            let start: String = dict.get_item("start_time").unwrap().extract()?;
            let end: String = dict.get_item("end_time").unwrap().extract()?;

            let start_time = NaiveTime::parse_from_str(&start, "%H:%M")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

            let end_time = NaiveTime::parse_from_str(&end, "%H:%M")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

            classes.push(ClassRequest {
                class_name,
                section,
                subject,
                start_time,
                end_time,
            });
        }

        // --------------------------
        // Use your updated scheduler
        // --------------------------
        //
        // Choose embedding dimension (32 or 64 recommended)
        //
        let embedding_dim = 32;

        let mut scheduler = FullScheduler::new(embedding_dim);

        // Compute schedule
        let results = scheduler.assign_classes(&classes);

        // Convert Rust output → Python list of dicts
        let py_list = pyo3::types::PyList::empty(py);

        for r in results {
            let d = pyo3::types::PyDict::new(py);

            d.set_item("class_name", r.class.class_name)?;
            d.set_item("section", r.class.section)?;
            d.set_item("subject", r.class.subject)?;
            d.set_item("teacher_id", r.teacher.id)?;
            d.set_item("teacher_name", r.teacher.name)?;
            d.set_item("similarity", r.teacher.similarity)?;
            d.set_item("reliability", r.teacher.reliability)?;
            d.set_item("workload", r.teacher.workload)?;

            py_list.append(d)?;
        }

        Ok(py_list.to_object(py))
    })
}

#[pyfunction]
pub fn schedule_classes_beam(py_classes: Vec<PyObject>) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // -------------------------
        // Convert Python → Rust
        // -------------------------
        let mut classes = Vec::new();

        for cls in py_classes {
            let dict: &pyo3::types::PyDict = cls.extract(py)?;

            let class_name: String = dict.get_item("class_name").unwrap().extract()?;
            let section: String = dict.get_item("section").unwrap().extract()?;
            let subject: String = dict.get_item("subject").unwrap().extract()?;

            let start: String = dict.get_item("start_time").unwrap().extract()?;
            let end: String = dict.get_item("end_time").unwrap().extract()?;

            let start_time = NaiveTime::parse_from_str(&start, "%H:%M")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

            let end_time = NaiveTime::parse_from_str(&end, "%H:%M")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

            classes.push(ClassRequest {
                class_name,
                section,
                subject,
                start_time,
                end_time,
            });
        }

        // --------------------------------------
        // Initialise your Beam Search Scheduler
        // --------------------------------------
        let embedding_dim = 32;

        let mut scheduler = GraphScheduler::new(
            embedding_dim,
            60,   // beam width (increase for higher accuracy)
            0.02, // similarity threshold
        );

        // Run beam scheduler
        let results = scheduler.assign_classes_beam(&classes);

        // -------------------------
        // Convert Rust → Python
        // -------------------------
        let py_list = pyo3::types::PyList::empty(py);

        for r in results {
            let d = pyo3::types::PyDict::new(py);

            d.set_item("class_name", r.class.class_name)?;
            d.set_item("section", r.class.section)?;
            d.set_item("subject", r.class.subject)?;

            d.set_item("teacher_id", r.teacher.id)?;
            d.set_item("teacher_name", r.teacher.name)?;

            d.set_item("similarity", r.teacher.similarity)?;
            d.set_item("reliability", r.teacher.reliability)?;
            d.set_item("workload", r.teacher.workload)?;

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