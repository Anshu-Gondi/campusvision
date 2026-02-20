use pyo3::prelude::*;
use crate::scheduler::python::scheduler::register;

mod scheduler;

#[pymodule]
fn intelligence_py(_py: Python, m: &PyModule) -> PyResult<()> {
    register(m)?;
    Ok(())
}