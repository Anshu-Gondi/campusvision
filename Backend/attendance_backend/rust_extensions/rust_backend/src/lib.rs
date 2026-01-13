// src/lib.rs
#![allow(unsafe_op_in_unsafe_fn)]

pub mod cctv_state;
pub mod cctv_tracker;
pub mod hnsw_helper;
pub mod models;
pub mod preprocess;
pub mod scheduler;
pub mod utils;

pub mod rust_only;
pub mod py_functions;

use pyo3::prelude::*;
use py_functions::add_functions;

#[pymodule]
fn rust_backend(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    add_functions(m)?;
    Ok(())
}
