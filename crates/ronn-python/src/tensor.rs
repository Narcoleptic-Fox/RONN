//! Tensor class for Python bindings

use pyo3::prelude::*;
use ronn_core::Tensor;

/// Tensor wrapper for Python
#[pyclass(name = "Tensor")]
pub struct PyTensor {
    pub(crate) inner: Tensor,
}

impl PyTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self { inner: tensor }
    }
}

#[pymethods]
impl PyTensor {
    /// Get tensor shape
    fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    /// Get tensor data type
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }

    /// Get number of elements
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, numel={})",
            self.shape(),
            self.dtype(),
            self.numel()
        )
    }
}
