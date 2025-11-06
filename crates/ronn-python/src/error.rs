//! Error handling for Python bindings

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

/// Python-facing error type
#[derive(Debug)]
pub struct RonnError(pub String);

impl From<ronn_core::CoreError> for RonnError {
    fn from(err: ronn_core::CoreError) -> Self {
        RonnError(err.to_string())
    }
}

impl From<ronn_api::Error> for RonnError {
    fn from(err: ronn_api::Error) -> Self {
        RonnError(err.to_string())
    }
}

impl From<anyhow::Error> for RonnError {
    fn from(err: anyhow::Error) -> Self {
        RonnError(err.to_string())
    }
}

impl From<RonnError> for PyErr {
    fn from(err: RonnError) -> PyErr {
        PyException::new_err(err.0)
    }
}

pub type PyResult<T> = Result<T, RonnError>;
