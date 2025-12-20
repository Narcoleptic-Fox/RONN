//! Model class for Python bindings

use crate::error::{PyResult, RonnError};
use crate::session::PySession;
use pyo3::prelude::*;
use ronn_api::Model;

/// ONNX model
///
/// # Example
///
/// ```python
/// model = ronn.Model.load("model.onnx")
/// print(f"Inputs: {model.input_names()}")
/// print(f"Outputs: {model.output_names()}")
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
    fn input_names(&self) -> Vec<String> {
        self.inner
            .input_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get model output names
    ///
    /// # Returns
    ///
    /// List of output tensor names
    fn output_names(&self) -> Vec<String> {
        self.inner
            .output_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Create an inference session
    ///
    /// # Arguments
    ///
    /// * `optimization_level` - Graph optimization level ("none", "basic", "aggressive")
    /// * `provider` - Execution provider ("cpu", "gpu", "bitnet", "wasm")
    /// * `num_threads` - Number of threads for CPU execution
    ///
    /// # Example
    ///
    /// ```python
    /// session = model.create_session(
    ///     optimization_level="aggressive",
    ///     provider="cpu",
    ///     num_threads=4
    /// )
    /// ```
    #[pyo3(signature = (optimization_level="basic", provider="cpu", num_threads=None))]
    fn create_session(
        &self,
        optimization_level: &str,
        provider: &str,
        num_threads: Option<usize>,
    ) -> PyResult<PySession> {
        use ronn_api::SessionOptions;

        // Parse optimization level
        let opt_level = match optimization_level {
            "O0" | "none" => ronn_graph::OptimizationLevel::O0,
            "O1" | "basic" => ronn_graph::OptimizationLevel::O1,
            "O2" | "default" => ronn_graph::OptimizationLevel::O2,
            "O3" | "aggressive" => ronn_graph::OptimizationLevel::O3,
            _ => {
                return Err(RonnError(format!(
                    "Invalid optimization level: {}. Use 'none'/'O0', 'basic'/'O1', 'default'/'O2', or 'aggressive'/'O3'",
                    optimization_level
                )));
            }
        };

        // Parse provider
        let provider_type = match provider {
            "cpu" => ronn_core::ProviderId::CPU,
            "gpu" | "cuda" => ronn_core::ProviderId::GPU,
            "bitnet" => ronn_core::ProviderId::BitNet,
            "wasm" => ronn_core::ProviderId::WebAssembly,
            _ => {
                return Err(RonnError(format!(
                    "Invalid provider: {}. Use 'cpu', 'gpu', 'bitnet', or 'wasm'",
                    provider
                )));
            }
        };

        // Build session options
        let mut options = SessionOptions::default()
            .with_optimization_level(opt_level)
            .with_provider(provider_type);

        if let Some(threads) = num_threads {
            options = options.with_num_threads(threads);
        }

        // Create session
        let session = self
            .inner
            .create_session(options)
            .map_err(RonnError::from)?;

        Ok(PySession::new(session))
    }

    /// Get model metadata
    ///
    /// # Returns
    ///
    /// Dictionary with model information
    fn metadata(&self, py: Python) -> PyObject {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("inputs", self.input_names()).unwrap();
        dict.set_item("outputs", self.output_names()).unwrap();
        dict.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(inputs={:?}, outputs={:?})",
            self.input_names(),
            self.output_names()
        )
    }
}
