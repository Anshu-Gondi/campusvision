// src/py_functions/mod.rs

pub mod face_recognition;
pub mod detection;
pub mod cctv;
pub mod database;
pub mod scheduler;
pub mod utils;

use pyo3::prelude::*;

// This function will be called from lib.rs
pub fn add_functions(m: &PyModule) -> PyResult<()> {
    face_recognition::register(m)?;
    detection::register(m)?;
    cctv::register(m)?;
    database::register(m)?;
    scheduler::register(m)?;
    utils::register(m)?;
    Ok(())
}