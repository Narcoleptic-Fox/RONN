//! Model class for Python bindings

use crate::error::{PyResult, RonnError};
use crate::session::PySession;
use crate::{PyOptimizationLevel, PyProviderType};
use pyo3::prelude::*;
use ronn_api::Model;

/// ONNX model
///
/// # Example
///
/// ```python
/// model = ronn.Model.load("model.onnx")
/// print(f"Inputs: {model.inputs()}")
/// print(f"Outputs: {model.outputs()}")
/// ```
#[pyclass(name = "Model")]
pub struct PyModel {
    inner: Model,
}

#[pymethods]
impl PyModel {
    /// Load a model from an ONNX file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .onnx file
    ///
    /// # Example
    ///
    /// ```python
    /// model = ronn.Model.load("resnet18.onnx")
    /// ```
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let model = Model::load(path).map_err(RonnError::from)?;
        Ok(Self { inner: model })
    }

    /// Get model input names
    ///
    /// # Returns
    ///
    /// List of input tensor names
    fn inputs(&self) -> Vec<String> {
        self.inner.inputs()
    }

    /// Get model output names
    ///
    /// # Returns
    ///
    /// List of output tensor names
    fn outputs(&self) -> Vec<String> {
        self.inner.outputs()
    }

    /// Create an inference session
    ///
    /// # Arguments
    ///
    /// * `optimization_level` - Graph optimization level ("O0", "O1", "O2", "O3")
    /// * `provider` - Execution provider ("cpu", "gpu", "bitnet", "wasm")
    /// * `num_threads` - Number of threads for CPU execution
    ///
    /// # Example
    ///
    /// ```python
    /// session = model.create_session(
    ///     optimization_level="O3",
    ///     provider="cpu",
    ///     num_threads=4
    /// )
    /// ```
    #[pyo3(signature = (optimization_level="O2", provider="cpu", num_threads=None))]
    fn create_session(
        &self,
        optimization_level: &str,
        provider: &str,
        num_threads: Option<usize>,
    ) -> PyResult<PySession> {
        use ronn_api::{SessionBuilder, SessionOptions};

        // Parse optimization level
        let opt_level = match optimization_level {
            "O0" | "none" => ronn_core::OptimizationLevel::O0,
            "O1" | "basic" => ronn_core::OptimizationLevel::O1,
            "O2" | "default" => ronn_core::OptimizationLevel::O2,
            "O3" | "aggressive" => ronn_core::OptimizationLevel::O3,
            _ => return Err(RonnError(format!("Invalid optimization level: {}", optimization_level))),
        };

        // Parse provider
        let provider_type = match provider {
            "cpu" => ronn_providers::ProviderType::Cpu,
            "gpu" | "cuda" => ronn_providers::ProviderType::Gpu,
            "bitnet" => ronn_providers::ProviderType::BitNet,
            "wasm" => ronn_providers::ProviderType::Wasm,
            _ => return Err(RonnError(format!("Invalid provider: {}", provider))),
        };

        // Build session options
        let mut options = SessionOptions::default()
            .with_optimization_level(opt_level)
            .with_provider(provider_type);

        if let Some(threads) = num_threads {
            options = options.with_num_threads(threads);
        }

        // Create session
        let session = self.inner.create_session(options).map_err(RonnError::from)?;

        Ok(PySession::new(session))
    }

    /// Get model metadata
    ///
    /// # Returns
    ///
    /// Dictionary with model information
    fn metadata(&self) -> pyo3::types::PyDict {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("inputs", self.inputs()).unwrap();
            dict.set_item("outputs", self.outputs()).unwrap();
            dict
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(inputs={:?}, outputs={:?})",
            self.inputs(),
            self.outputs()
        )
    }
}
