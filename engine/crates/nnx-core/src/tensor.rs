//! Core tensor type for the NNX engine.
//!
//! This is a *storage-level* tensor — it holds raw data with shape and dtype
//! metadata. Compute operations live in `nnx-kernels`, not here. This keeps
//! the core type lightweight and avoids coupling storage to any specific
//! compute backend.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{EngineError, Result};
use crate::shape::Shape;
use std::sync::Arc;

/// Owned tensor with shape, dtype, and device metadata.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Raw data as bytes. Interpretation depends on `dtype`.
    data: Arc<Vec<u8>>,
    /// Shape of the tensor.
    shape: Shape,
    /// Element data type.
    dtype: DType,
    /// Device this tensor lives on.
    device: Device,
}

impl Tensor {
    /// Create a tensor from raw bytes.
    ///
    /// # Errors
    /// Returns an error if data length doesn't match shape * dtype size.
    pub fn from_raw(data: Vec<u8>, shape: Shape, dtype: DType, device: Device) -> Result<Self> {
        let expected = shape.numel() * dtype.size_bytes();
        if data.len() != expected {
            return Err(EngineError::ShapeMismatch(format!(
                "expected {} bytes for shape {} with dtype {}, got {}",
                expected,
                shape,
                dtype,
                data.len()
            )));
        }
        Ok(Self {
            data: Arc::new(data),
            shape,
            dtype,
            device,
        })
    }

    /// Create a tensor from an f32 slice.
    pub fn from_f32(data: &[f32], shape: Shape) -> Result<Self> {
        let expected = shape.numel();
        if data.len() != expected {
            return Err(EngineError::ShapeMismatch(format!(
                "expected {} elements for shape {}, got {}",
                expected,
                shape,
                data.len()
            )));
        }
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::from_raw(bytes, shape, DType::F32, Device::Cpu)
    }

    /// Create a zero-filled tensor.
    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        let size = shape.numel() * dtype.size_bytes();
        Self {
            data: Arc::new(vec![0u8; size]),
            shape,
            dtype,
            device: Device::Cpu,
        }
    }

    /// Shape of this tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Data type of elements.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device this tensor lives on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Raw byte data.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Interpret data as f32 slice. Panics if dtype is not F32.
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "tensor is not f32");
        // SAFETY: data was created from f32s and alignment is guaranteed by Vec<u8>
        // on little-endian systems. For production, use bytemuck.
        let ptr = self.data.as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.numel()) }
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Borrowed view into a tensor's data (zero-copy).
#[derive(Debug)]
pub struct TensorView<'a> {
    data: &'a [u8],
    shape: Shape,
    dtype: DType,
}

impl<'a> TensorView<'a> {
    /// Create a view from raw byte data.
    pub fn new(data: &'a [u8], shape: Shape, dtype: DType) -> Result<Self> {
        let expected = shape.numel() * dtype.size_bytes();
        if data.len() < expected {
            return Err(EngineError::ShapeMismatch(format!(
                "view needs {} bytes, got {}",
                expected,
                data.len()
            )));
        }
        Ok(Self { data, shape, dtype })
    }

    pub fn shape(&self) -> &Shape { &self.shape }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn as_bytes(&self) -> &[u8] { self.data }

    /// Copy this view into an owned tensor.
    pub fn to_tensor(&self, device: Device) -> Tensor {
        Tensor {
            data: Arc::new(self.data.to_vec()),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_f32(&data, Shape::new(&[2, 3])).unwrap();
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.as_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = Tensor::from_f32(&data, Shape::new(&[2, 3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(Shape::new(&[4, 4]), DType::F32);
        assert_eq!(t.numel(), 16);
        assert!(t.as_f32().iter().all(|&v| v == 0.0));
    }
}
