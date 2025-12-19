//! Session class for Python bindings

use crate::error::{PyResult, RonnError};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use ronn_api::InferenceSession;
use std::collections::HashMap;

/// Inference session
///
/// # Example
///
/// ```python
/// import numpy as np
///
/// inputs = {
///     "input": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
/// }
/// outputs = session.run(inputs)
/// print(outputs["output"])
/// ```
#[pyclass(name = "Session")]
pub struct PySession {
    inner: InferenceSession,
}

impl PySession {
    pub fn new(session: InferenceSession) -> Self {
        Self { inner: session }
    }
}

#[pymethods]
impl PySession {
    /// Run inference
    ///
    /// # Arguments
    ///
    /// * `inputs` - Dictionary mapping input names to numpy arrays
    ///
    /// # Returns
    ///
    /// Dictionary mapping output names to numpy arrays
    ///
    /// # Example
    ///
    /// ```python
    /// outputs = session.run({"input": input_array})
    /// result = outputs["output"]
    /// ```
    fn run(&self, py: Python, inputs: &PyDict) -> PyResult<PyObject> {
        // Convert Python dict to HashMap<String, Tensor>
        let mut input_tensors: HashMap<String, ronn_core::Tensor> = HashMap::new();

        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;

            // Handle different numpy array types
            let tensor = if let Ok(array) = value.downcast::<PyArray1<f32>>() {
                let data: Vec<f32> = array.to_vec()?;
                let shape = vec![array.len()];
                ronn_core::Tensor::from_data(
                    data,
                    shape,
                    ronn_core::DataType::F32,
                    ronn_core::TensorLayout::RowMajor,
                )
                .map_err(RonnError::from)?
            } else {
                return Err(RonnError(format!(
                    "Unsupported input type for '{}'",
                    name
                )));
            };

            input_tensors.insert(name, tensor);
        }

        // Convert HashMap<String, Tensor> to HashMap<&str, Tensor>
        let inputs_ref: HashMap<&str, ronn_core::Tensor> = input_tensors
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();

        // Run inference
        let output_tensors = self.inner.run(inputs_ref).map_err(RonnError::from)?;

        // Convert back to Python dict
        let result = PyDict::new(py);
        for (name, tensor) in output_tensors {
            // Convert tensor to numpy array
            let data: Vec<f32> = tensor.to_vec().unwrap_or_else(|_| vec![]);
            let array = PyArray1::from_vec(py, data);
            result.set_item(name, array)?;
        }

        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        "Session(...)".to_string()
    }
}
