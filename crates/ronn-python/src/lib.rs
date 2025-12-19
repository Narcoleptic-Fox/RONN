//! Python bindings for RONN
//!
//! Provides a Pythonic interface to RONN's high-performance ONNX inference runtime.
//!
//! # Installation
//!
//! ```bash
//! pip install ronn
//! ```
//!
//! # Quick Start
//!
//! ```python
//! import ronn
//! import numpy as np
//!
//! # Load model
//! model = ronn.Model.load("model.onnx")
//!
//! # Create session
//! session = model.create_session(
//!     optimization_level="O3",
//!     provider="cpu"
//! )
//!
//! # Run inference
//! inputs = {"input": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}
//! outputs = session.run(inputs)
//! print(outputs["output"])
//! ```

use pyo3::prelude::*;

mod error;
mod model;
mod session;
mod tensor;

pub use error::RonnError;
pub use model::PyModel;
pub use session::PySession;
pub use tensor::PyTensor;

/// RONN Python module
#[pymodule]
fn ronn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PySession>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyOptimizationLevel>()?;
    m.add_class::<PyProviderType>()?;
    m.add_class::<PyBatchConfig>()?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// Optimization level for graph optimizations
#[pyclass(name = "OptimizationLevel")]
#[derive(Clone)]
pub struct PyOptimizationLevel {
    inner: ronn_core::OptimizationLevel,
}

#[pymethods]
impl PyOptimizationLevel {
    /// No optimizations (O0)
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: ronn_core::OptimizationLevel::None,
        }
    }

    /// Basic optimizations (O1)
    #[staticmethod]
    fn basic() -> Self {
        Self {
            inner: ronn_core::OptimizationLevel::Basic,
        }
    }

    /// Default optimizations (O2, recommended)
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: ronn_core::OptimizationLevel::Basic,
        }
    }

    /// Aggressive optimizations (O3)
    #[staticmethod]
    fn aggressive() -> Self {
        Self {
            inner: ronn_core::OptimizationLevel::Aggressive,
        }
    }
}

impl Default for PyOptimizationLevel {
    fn default() -> Self {
        Self::default()
    }
}

/// Execution provider type
#[pyclass(name = "ProviderType")]
#[derive(Clone)]
pub struct PyProviderType {
    inner: ronn_core::ProviderId,
}

#[pymethods]
impl PyProviderType {
    /// CPU provider (default)
    #[staticmethod]
    fn cpu() -> Self {
        Self {
            inner: ronn_core::ProviderId::CPU,
        }
    }

    /// GPU provider (CUDA)
    #[staticmethod]
    fn gpu() -> Self {
        Self {
            inner: ronn_core::ProviderId::GPU,
        }
    }

    /// BitNet provider (1.58-bit quantization)
    #[staticmethod]
    fn bitnet() -> Self {
        Self {
            inner: ronn_core::ProviderId::BitNet,
        }
    }

    /// WebAssembly provider
    #[staticmethod]
    fn wasm() -> Self {
        Self {
            inner: ronn_core::ProviderId::WebAssembly,
        }
    }
}

impl Default for PyProviderType {
    fn default() -> Self {
        Self::cpu()
    }
}

/// Batch processing configuration
#[pyclass(name = "BatchConfig")]
#[derive(Clone)]
pub struct PyBatchConfig {
    #[pyo3(get, set)]
    /// Maximum batch size
    pub max_batch_size: usize,

    #[pyo3(get, set)]
    /// Timeout in milliseconds
    pub timeout_ms: u64,

    #[pyo3(get, set)]
    /// Queue capacity
    pub queue_capacity: usize,
}

#[pymethods]
impl PyBatchConfig {
    #[new]
    #[pyo3(signature = (max_batch_size=32, timeout_ms=10, queue_capacity=1024))]
    fn new(max_batch_size: usize, timeout_ms: u64, queue_capacity: usize) -> Self {
        Self {
            max_batch_size,
            timeout_ms,
            queue_capacity,
        }
    }
}

impl Default for PyBatchConfig {
    fn default() -> Self {
        Self::new(32, 10, 1024)
    }
}
