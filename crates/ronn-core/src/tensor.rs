//! Tensor implementation with Candle backend integration.
//!
//! This module provides the core Tensor type for RONN with seamless integration
//! to the Candle tensor library for high-performance operations and GPU acceleration.

use crate::ops::shape::ShapeOps;
use crate::types::{DataType, Tensor as RonnTensor, TensorLayout};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Module, Shape, Tensor as CandleTensor};

/// Enhanced Tensor implementation with Candle backend.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The underlying Candle tensor for computation.
    candle_tensor: CandleTensor,
    /// Original data type specification.
    dtype: DataType,
    /// Memory layout preference.
    layout: TensorLayout,
}

impl Tensor {
    /// Create a new tensor from raw data.
    ///
    /// # Arguments
    /// * `data` - Raw tensor data
    /// * `shape` - Tensor dimensions
    /// * `dtype` - Data type specification
    /// * `layout` - Memory layout preference
    ///
    /// # Example
    /// ```rust
    /// use ronn_core::tensor::Tensor;
    /// use ronn_core::types::{DataType, TensorLayout};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0];
    /// let tensor = Tensor::from_data(data, vec![2, 2], DataType::F32, TensorLayout::RowMajor)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_data(
        data: Vec<f32>,
        shape: Vec<usize>,
        dtype: DataType,
        layout: TensorLayout,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let candle_shape = Shape::from_dims(&shape);

        let candle_tensor = match dtype {
            DataType::F32 => CandleTensor::from_vec(data, candle_shape, &device)?,
            DataType::F16 => {
                let f16_data: Vec<half::f16> = data.into_iter().map(half::f16::from_f32).collect();
                CandleTensor::from_vec(f16_data, candle_shape, &device)?
            }
            DataType::BF16 => {
                let bf16_data: Vec<half::bf16> =
                    data.into_iter().map(half::bf16::from_f32).collect();
                CandleTensor::from_vec(bf16_data, candle_shape, &device)?
            }
            DataType::F64 => {
                let f64_data: Vec<f64> = data.into_iter().map(|x| x as f64).collect();
                CandleTensor::from_vec(f64_data, candle_shape, &device)?
            }
            DataType::U8 => {
                let u8_data: Vec<u8> = data.into_iter().map(|x| x as u8).collect();
                CandleTensor::from_vec(u8_data, candle_shape, &device)?
            }
            DataType::U32 => {
                let u32_data: Vec<u32> = data.into_iter().map(|x| x as u32).collect();
                CandleTensor::from_vec(u32_data, candle_shape, &device)?
            }
            // For unsupported types, convert to F32
            DataType::I8 | DataType::I32 | DataType::I64 | DataType::Bool => {
                CandleTensor::from_vec(data, candle_shape, &device)?
            }
        };

        Ok(Self {
            candle_tensor,
            dtype,
            layout,
        })
    }

    /// Create an INT64 tensor from raw i64 data.
    pub fn from_i64(data: Vec<i64>, shape: Vec<usize>, layout: TensorLayout) -> Result<Self> {
        let device = Device::Cpu;
        let candle_shape = Shape::from_dims(&shape);
        let candle_tensor = CandleTensor::from_vec(data, candle_shape, &device)?;
        Ok(Self {
            candle_tensor,
            dtype: DataType::I64,
            layout,
        })
    }

    /// Create an INT32 tensor from raw i32 data.
    pub fn from_i32(data: Vec<i32>, shape: Vec<usize>, layout: TensorLayout) -> Result<Self> {
        let device = Device::Cpu;
        let candle_shape = Shape::from_dims(&shape);
        let candle_tensor = CandleTensor::from_vec(data, candle_shape, &device)?;
        Ok(Self {
            candle_tensor,
            dtype: DataType::I32,
            layout,
        })
    }

    /// Create a tensor filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - Tensor dimensions
    /// * `dtype` - Data type specification
    /// * `layout` - Memory layout preference
    pub fn zeros(shape: Vec<usize>, dtype: DataType, layout: TensorLayout) -> Result<Self> {
        let device = Device::Cpu;
        let candle_dtype = dtype_to_candle(&dtype)?;
        let candle_shape = Shape::from_dims(&shape);

        let candle_tensor = CandleTensor::zeros(candle_shape, candle_dtype, &device)?;

        Ok(Self {
            candle_tensor,
            dtype,
            layout,
        })
    }

    /// Create a tensor filled with ones.
    ///
    /// # Arguments
    /// * `shape` - Tensor dimensions
    /// * `dtype` - Data type specification
    /// * `layout` - Memory layout preference
    pub fn ones(shape: Vec<usize>, dtype: DataType, layout: TensorLayout) -> Result<Self> {
        let device = Device::Cpu;
        let candle_dtype = dtype_to_candle(&dtype)?;
        let candle_shape = Shape::from_dims(&shape);

        let candle_tensor = CandleTensor::ones(candle_shape, candle_dtype, &device)?;

        Ok(Self {
            candle_tensor,
            dtype,
            layout,
        })
    }

    /// Create a tensor with random values from a uniform distribution.
    pub fn rand(shape: Vec<usize>, dtype: DataType, layout: TensorLayout) -> Result<Self> {
        let device = Device::Cpu;
        let _candle_dtype = dtype_to_candle(&dtype)?;
        let candle_shape = Shape::from_dims(&shape);

        let candle_tensor = CandleTensor::rand(0.0, 1.0, candle_shape, &device)?;

        Ok(Self {
            candle_tensor,
            dtype,
            layout,
        })
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.candle_tensor.dims().to_vec()
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get the memory layout of the tensor.
    pub fn layout(&self) -> TensorLayout {
        self.layout
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.candle_tensor.dims().len()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.candle_tensor.elem_count()
    }

    /// Get the device where the tensor is stored.
    pub fn device(&self) -> &Device {
        self.candle_tensor.device()
    }

    /// Convert tensor to CPU device.
    pub fn to_cpu(&self) -> Result<Self> {
        let cpu_tensor = self.candle_tensor.to_device(&Device::Cpu)?;
        Ok(Self {
            candle_tensor: cpu_tensor,
            dtype: self.dtype,
            layout: self.layout,
        })
    }

    /// Convert tensor to GPU device (if available).
    #[cfg(feature = "gpu")]
    pub fn to_gpu(&self, device_id: usize) -> Result<Self> {
        let gpu_device = Device::new_cuda(device_id)?;
        let gpu_tensor = self.candle_tensor.to_device(&gpu_device)?;
        Ok(Self {
            candle_tensor: gpu_tensor,
            dtype: self.dtype,
            layout: self.layout,
        })
    }

    /// Extract data as a vector of f32 values.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        // Flatten the tensor first if it's multi-dimensional
        let flattened = if self.candle_tensor.dims().len() > 1 {
            self.candle_tensor.flatten_all()?
        } else {
            self.candle_tensor.clone()
        };

        match self.dtype {
            DataType::F32 => {
                let data: Vec<f32> = flattened.to_vec1()?;
                Ok(data)
            }
            DataType::I8 => {
                let data: Vec<f32> = flattened.to_vec1()?;
                Ok(data)
            }
            DataType::I32 => {
                let data: Vec<i32> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
            DataType::I64 => {
                let data: Vec<i64> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
            DataType::Bool => {
                let data: Vec<u8> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
            DataType::F16 => {
                let data: Vec<half::f16> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x.to_f32()).collect())
            }
            DataType::BF16 => {
                let data: Vec<half::bf16> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x.to_f32()).collect())
            }
            DataType::F64 => {
                let data: Vec<f64> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
            DataType::U8 => {
                let data: Vec<u8> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
            DataType::U32 => {
                let data: Vec<u32> = flattened.to_vec1()?;
                Ok(data.into_iter().map(|x| x as f32).collect())
            }
        }
    }

    /// Get the underlying Candle tensor for advanced operations.
    pub fn candle_tensor(&self) -> &CandleTensor {
        &self.candle_tensor
    }

    /// Create a Tensor from a Candle tensor.
    pub fn from_candle(candle_tensor: CandleTensor, dtype: DataType, layout: TensorLayout) -> Self {
        Self {
            candle_tensor,
            dtype,
            layout,
        }
    }

    /// Check if tensor shapes are broadcastable.
    pub fn is_broadcastable_with(&self, other: &Tensor) -> bool {
        let shape1 = self.shape();
        let shape2 = other.shape();

        // Pad shorter shape with 1s on the left
        let max_len = shape1.len().max(shape2.len());
        let mut padded1 = vec![1; max_len - shape1.len()];
        let mut padded2 = vec![1; max_len - shape2.len()];
        padded1.extend(shape1);
        padded2.extend(shape2);

        // Check compatibility dimension by dimension
        for (d1, d2) in padded1.iter().zip(padded2.iter()) {
            if *d1 != *d2 && *d1 != 1 && *d2 != 1 {
                return false;
            }
        }
        true
    }

    /// Compute broadcast shape for two tensors.
    pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
        let max_len = shape1.len().max(shape2.len());
        let mut padded1 = vec![1; max_len - shape1.len()];
        let mut padded2 = vec![1; max_len - shape2.len()];
        padded1.extend(shape1);
        padded2.extend(shape2);

        let mut result = Vec::with_capacity(max_len);
        for (d1, d2) in padded1.iter().zip(padded2.iter()) {
            match (d1, d2) {
                (1, d) | (d, 1) => result.push(*d),
                (d1, d2) if d1 == d2 => result.push(*d1),
                (d1, d2) => {
                    return Err(anyhow!(
                        "Cannot broadcast shapes: dimension {} vs {}",
                        d1,
                        d2
                    ));
                }
            }
        }
        Ok(result)
    }

    /// Convolution 2D operation (placeholder - uses Candle's conv2d).
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        strides: &[usize],
        pads: &[usize],
        dilations: &[usize],
        groups: usize,
    ) -> Result<Tensor> {
        // Simplified implementation - full implementation would use candle_nn
        let _ = (weight, bias, strides, pads, dilations, groups);
        Err(anyhow!("conv2d not yet fully implemented"))
    }

    /// Max pooling 2D operation.
    pub fn max_pool2d(
        &self,
        kernel_shape: &[usize],
        strides: &[usize],
        pads: &[usize],
    ) -> Result<Tensor> {
        let _ = (kernel_shape, strides, pads);
        Err(anyhow!("max_pool2d not yet fully implemented"))
    }

    /// Average pooling 2D operation.
    pub fn avg_pool2d(
        &self,
        kernel_shape: &[usize],
        strides: &[usize],
        pads: &[usize],
    ) -> Result<Tensor> {
        let _ = (kernel_shape, strides, pads);
        Err(anyhow!("avg_pool2d not yet fully implemented"))
    }

    /// Batch normalization operation.
    pub fn batch_norm(
        &self,
        scale: &Tensor,
        bias: &Tensor,
        mean: &Tensor,
        var: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor> {
        let _ = (scale, bias, mean, var, epsilon);
        Err(anyhow!("batch_norm not yet fully implemented"))
    }

    /// Get rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.ndim()
    }

    /// Convert to 1D vector (alias for to_vec).
    pub fn to_vec1<T: candle_core::WithDType>(&self) -> Result<Vec<T>> {
        let flattened = if self.candle_tensor.dims().len() > 1 {
            self.candle_tensor.flatten_all()?
        } else {
            self.candle_tensor.clone()
        };
        Ok(flattened.to_vec1()?)
    }

    /// Stack tensors along a new dimension.
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to stack
    /// * `dim` - Dimension along which to stack
    ///
    /// # Example
    /// ```rust
    /// use ronn_core::tensor::Tensor;
    /// use ronn_core::types::{DataType, TensorLayout};
    ///
    /// let t1 = Tensor::from_data(vec![1.0, 2.0], vec![2], DataType::F32, TensorLayout::RowMajor)?;
    /// let t2 = Tensor::from_data(vec![3.0, 4.0], vec![2], DataType::F32, TensorLayout::RowMajor)?;
    /// let stacked = Tensor::stack(&[&t1, &t2], 0)?;
    /// assert_eq!(stacked.shape(), vec![2, 2]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn stack(tensors: &[&Tensor], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(anyhow!("Cannot stack empty tensor list"));
        }

        let candle_tensors: Vec<_> = tensors.iter().map(|t| &t.candle_tensor).collect();
        let stacked = CandleTensor::stack(&candle_tensors, dim)?;

        Ok(Self {
            candle_tensor: stacked,
            dtype: tensors[0].dtype,
            layout: tensors[0].layout,
        })
    }

    /// Split tensor into chunks along an axis.
    ///
    /// # Arguments
    /// * `num_chunks` - Number of chunks to split into
    /// * `dim` - Dimension along which to split
    ///
    /// # Example
    /// ```rust
    /// use ronn_core::tensor::Tensor;
    /// use ronn_core::types::{DataType, TensorLayout};
    ///
    /// let t = Tensor::from_data(
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![2, 2],
    ///     DataType::F32,
    ///     TensorLayout::RowMajor
    /// )?;
    /// let chunks = t.split(2, 0)?;
    /// assert_eq!(chunks.len(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn split(&self, num_chunks: usize, dim: usize) -> Result<Vec<Tensor>> {
        if num_chunks == 0 {
            return Err(anyhow!("Cannot split into 0 chunks"));
        }

        let shape = self.shape();
        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} out of bounds for shape {:?}",
                dim,
                shape
            ));
        }

        let dim_size = shape[dim];
        if dim_size % num_chunks != 0 {
            return Err(anyhow!(
                "Dimension size {} not evenly divisible by {} chunks",
                dim_size,
                num_chunks
            ));
        }

        let chunk_size = dim_size / num_chunks;
        let mut chunks = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let start = i * chunk_size;
            let _end = start + chunk_size;
            let chunk = self.candle_tensor.narrow(dim, start, chunk_size)?;
            chunks.push(Self {
                candle_tensor: chunk,
                dtype: self.dtype,
                layout: self.layout,
            });
        }

        Ok(chunks)
    }

    /// Gather elements along an axis.
    pub fn gather(&self, indices: &Tensor, dim: usize) -> Result<Tensor> {
        let shape = self.shape();
        if dim >= shape.len() {
            return Err(anyhow!(
                "Dimension {} out of bounds for shape {:?}",
                dim,
                shape
            ));
        }

        // ONNX indices are typically int64, but we accept common integer types.
        let flat_indices: Vec<i64> = if let Ok(v) = indices.to_vec1::<i64>() {
            v
        } else if let Ok(v) = indices.to_vec1::<i32>() {
            v.into_iter().map(i64::from).collect()
        } else if let Ok(v) = indices.to_vec1::<u32>() {
            v.into_iter().map(i64::from).collect()
        } else if let Ok(v) = indices.to_vec1::<u8>() {
            v.into_iter().map(i64::from).collect()
        } else {
            return Err(anyhow!(
                "Gather indices must be an integer tensor, got {:?}",
                indices.dtype()
            ));
        };

        let indices_shape = indices.shape();
        let idx = CandleTensor::from_vec(
            flat_indices.clone(),
            vec![flat_indices.len()],
            self.candle_tensor.device(),
        )?;

        let selected = self.candle_tensor.index_select(&idx, dim)?;

        let mut out_shape = Vec::new();
        out_shape.extend_from_slice(&shape[..dim]);
        out_shape.extend(indices_shape.iter().copied());
        out_shape.extend_from_slice(&shape[dim + 1..]);

        let gathered = if out_shape == selected.dims() {
            selected
        } else {
            selected.reshape(out_shape)?
        };
        Ok(Tensor::from_candle(gathered, self.dtype, self.layout))
    }

    /// Transpose with specific permutation.
    pub fn transpose(&self, perm: &[usize]) -> Result<Tensor> {
        let result = self.candle_tensor.permute(perm)?;
        Ok(Tensor::from_candle(result, self.dtype, self.layout))
    }

    /// Layer normalization (critical for transformers).
    ///
    /// Normalizes the input across the specified axis.
    ///
    /// # Arguments
    /// * `scale` - Optional scale parameter (gamma)
    /// * `bias` - Optional bias parameter (beta)
    /// * `epsilon` - Small constant for numerical stability
    /// * `axis` - Axis to normalize over (default: -1 for last dimension)
    ///
    /// # Example
    /// ```ignore
    /// use ronn_core::tensor::Tensor;
    /// use ronn_core::types::{DataType, TensorLayout};
    ///
    /// let input = Tensor::from_data(
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![2, 2],
    ///     DataType::F32,
    ///     TensorLayout::RowMajor
    /// )?;
    /// let normalized = input.layer_norm(None, None, 1e-5, 1)?;  // Use positive axis
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn layer_norm(
        &self,
        scale: Option<&Tensor>,
        bias: Option<&Tensor>,
        epsilon: f32,
        axis: i32,
    ) -> Result<Self> {
        use candle_nn::LayerNorm;

        let shape = self.shape();
        let _normalized_shape = if axis == -1 {
            vec![shape[shape.len() - 1]]
        } else {
            let axis_usize = if axis < 0 {
                (shape.len() as i32 + axis) as usize
            } else {
                axis as usize
            };
            vec![shape[axis_usize]]
        };

        // Create layer norm config
        // If scale and bias provided, use them
        let normalized = if let (Some(s), Some(b)) = (scale, bias) {
            let ln = LayerNorm::new(
                s.candle_tensor.clone(),
                b.candle_tensor.clone(),
                epsilon as f64,
            );
            ln.forward(&self.candle_tensor)?
        } else {
            // Simple normalization without learnable parameters
            let mean = self.candle_tensor.mean_keepdim(axis as usize)?;
            let variance = self
                .candle_tensor
                .broadcast_sub(&mean)?
                .sqr()?
                .mean_keepdim(axis as usize)?;
            let std = (variance + epsilon as f64)?.sqrt()?;
            self.candle_tensor
                .broadcast_sub(&mean)?
                .broadcast_div(&std)?
        };

        Ok(Self::from_candle(normalized, self.dtype, self.layout))
    }

    /// Multi-head attention mechanism (critical for transformers).
    ///
    /// Computes scaled dot-product attention: softmax(Q·K^T / sqrt(d_k))·V
    ///
    /// # Arguments
    /// * `key` - Key tensor
    /// * `value` - Value tensor
    /// * `num_heads` - Number of attention heads
    /// * `mask` - Optional attention mask
    ///
    /// # Example
    /// ```rust
    /// use ronn_core::tensor::Tensor;
    /// use ronn_core::types::{DataType, TensorLayout};
    ///
    /// let query = Tensor::from_data(
    ///     vec![1.0; 64],
    ///     vec![1, 8, 8],  // (batch, seq_len, d_model)
    ///     DataType::F32,
    ///     TensorLayout::RowMajor
    /// )?;
    /// let key = query.clone();
    /// let value = query.clone();
    ///
    /// let output = query.attention(&key, &value, 2, None)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        num_heads: usize,
        mask: Option<&Tensor>,
    ) -> Result<Self> {
        let query = &self.candle_tensor;
        let key = &key.candle_tensor;
        let value = &value.candle_tensor;

        // Get dimensions
        let query_shape = query.dims();
        if query_shape.len() != 3 {
            return Err(anyhow!(
                "Query must be 3D (batch, seq_len, d_model), got {:?}",
                query_shape
            ));
        }

        let batch_size = query_shape[0];
        let seq_len = query_shape[1];
        let d_model = query_shape[2];

        if d_model % num_heads != 0 {
            return Err(anyhow!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model,
                num_heads
            ));
        }

        let d_k = d_model / num_heads;

        // Reshape Q, K, V for multi-head attention
        // (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        let q = query
            .reshape(&[batch_size, seq_len, num_heads, d_k])?
            .transpose(1, 2)?;
        let k = key
            .reshape(&[batch_size, seq_len, num_heads, d_k])?
            .transpose(1, 2)?;
        let v = value
            .reshape(&[batch_size, seq_len, num_heads, d_k])?
            .transpose(1, 2)?;

        // Compute attention scores: Q·K^T / sqrt(d_k)
        let k_t = k.transpose(2, 3)?;
        let scores = (q.matmul(&k_t)? / (d_k as f64).sqrt())?;

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            scores.broadcast_add(&m.candle_tensor)?
        } else {
            scores
        };

        // Apply softmax
        let attention_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Apply attention to values: attention_weights·V
        let output = attention_weights.matmul(&v)?;

        // Reshape back: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        let output = output
            .transpose(1, 2)?
            .reshape(&[batch_size, seq_len, d_model])?;

        Ok(Self::from_candle(output, self.dtype, self.layout))
    }

    /// Clip values to a range [min, max]
    pub fn clip(&self, min: f32, max: f32) -> Result<Self> {
        let result = self.candle_tensor.clamp(min, max)?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Element-wise power operation with tensor exponent.
    /// For scalar exponents, use the `ArithmeticOps::pow` trait method instead.
    pub fn pow_tensor(&self, exponent: &Tensor) -> Result<Self> {
        let result = self.candle_tensor.pow(&exponent.candle_tensor)?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Self> {
        let result = self.candle_tensor.sqrt()?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Element-wise exponential (e^x)
    pub fn exp(&self) -> Result<Self> {
        let result = self.candle_tensor.exp()?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> Result<Self> {
        let result = self.candle_tensor.log()?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Element-wise negation
    pub fn neg(&self) -> Result<Self> {
        let result = self.candle_tensor.neg()?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Result<Self> {
        let result = self.candle_tensor.abs()?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// LeakyReLU activation: max(alpha * x, x)
    pub fn leaky_relu(&self, alpha: f32) -> Result<Self> {
        let scaled = self.candle_tensor.affine(alpha as f64, 0.0)?;
        let result = self.candle_tensor.maximum(&scaled)?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// ELU activation: x if x > 0 else alpha * (exp(x) - 1)
    pub fn elu(&self, alpha: f32) -> Result<Self> {
        // ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
        let zero = self.candle_tensor.zeros_like()?;
        let mask = self.candle_tensor.gt(&zero)?;

        let positive_part = &self.candle_tensor;
        let exp_part = self.candle_tensor.exp()?.affine(1.0, -1.0)?;
        let negative_part = exp_part.affine(alpha as f64, 0.0)?;

        let result = mask.where_cond(positive_part, &negative_part)?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Swish/SiLU activation: x * sigmoid(x)
    pub fn swish(&self) -> Result<Self> {
        let sigmoid = candle_nn::ops::sigmoid(&self.candle_tensor)?;
        let result = (&self.candle_tensor * &sigmoid)?;
        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Remove dimensions of size 1
    pub fn squeeze(&self, axes: Option<Vec<usize>>) -> Result<Self> {
        let shape = self.shape();
        let new_shape: Vec<usize> = if let Some(axes) = axes {
            // Remove specific axes
            shape
                .iter()
                .enumerate()
                .filter(|(i, dim)| !axes.contains(i) || **dim != 1)
                .map(|(_, dim)| *dim)
                .collect()
        } else {
            // Remove all dimensions of size 1
            shape.iter().copied().filter(|dim| *dim != 1).collect()
        };

        if new_shape.is_empty() {
            // If all dimensions were 1, keep at least one
            return self.reshape(&[1]);
        }

        self.reshape(&new_shape)
    }

    /// Add dimensions of size 1
    pub fn unsqueeze(&self, axes: &[usize]) -> Result<Self> {
        let mut new_shape = self.shape();
        let mut axes_sorted = axes.to_vec();
        axes_sorted.sort_unstable();

        for &axis in &axes_sorted {
            // Check bounds before inserting
            if axis > new_shape.len() {
                return Err(anyhow!(
                    "Unsqueeze axis {} is out of bounds for shape with {} dimensions",
                    axis,
                    new_shape.len()
                ));
            }
            new_shape.insert(axis, 1);
        }

        self.reshape(&new_shape)
    }

    /// Reduce mean along axes
    pub fn reduce_mean(&self, axes: &[usize], keepdims: bool) -> Result<Self> {
        let mut result = self.candle_tensor.clone();

        // Sort axes in descending order to maintain correct indices
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            result = result.mean_keepdim(axis)?;
            if !keepdims {
                result = result.squeeze(axis)?;
            }
        }

        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Reduce sum along axes
    pub fn reduce_sum(&self, axes: &[usize], keepdims: bool) -> Result<Self> {
        let mut result = self.candle_tensor.clone();

        // Sort axes in descending order to maintain correct indices
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort_unstable_by(|a, b| b.cmp(a));

        for &axis in &sorted_axes {
            result = result.sum_keepdim(axis)?;
            if !keepdims {
                result = result.squeeze(axis)?;
            }
        }

        Ok(Self::from_candle(result, self.dtype, self.layout))
    }

    /// Cast tensor to a different data type
    pub fn cast(&self, to: DataType) -> Result<Self> {
        let target_dtype = dtype_to_candle(&to)?;
        let result = self.candle_tensor.to_dtype(target_dtype)?;
        Ok(Self::from_candle(result, to, self.layout))
    }

    /// Convert tensor to a scalar f32 value
    pub fn to_scalar_f32(&self) -> Result<f32> {
        let value = self.candle_tensor.to_scalar::<f32>()?;
        Ok(value)
    }
}

/// Convert RONN DataType to Candle DType.
fn dtype_to_candle(dtype: &DataType) -> Result<DType> {
    match dtype {
        DataType::F32 => Ok(DType::F32),
        DataType::F16 => Ok(DType::F16),
        DataType::BF16 => Ok(DType::BF16),
        DataType::F64 => Ok(DType::F64),
        DataType::U8 => Ok(DType::U8),
        DataType::U32 => Ok(DType::U32),
        // For unsupported types, use F32
        DataType::I8 | DataType::I32 | DataType::I64 | DataType::Bool => Ok(DType::F32),
    }
}

/// Convert Candle DType to RONN DataType.
#[allow(dead_code)]
fn dtype_from_candle(dtype: DType) -> DataType {
    match dtype {
        DType::F32 => DataType::F32,
        DType::F16 => DataType::F16,
        DType::U8 => DataType::U8,
        DType::U32 => DataType::U32,
        DType::F64 => DataType::F64,
        _ => DataType::F32, // Default fallback
    }
}

/// Convert legacy RonnTensor to new Tensor implementation.
impl From<RonnTensor> for Tensor {
    fn from(legacy: RonnTensor) -> Self {
        Self::from_data(legacy.data, legacy.shape, legacy.dtype, legacy.layout)
            .expect("Failed to convert legacy tensor")
    }
}

/// Convert new Tensor to legacy RonnTensor for compatibility.
impl From<Tensor> for RonnTensor {
    fn from(tensor: Tensor) -> Self {
        let data = tensor.to_vec().expect("Failed to extract tensor data");
        Self {
            data,
            shape: tensor.shape(),
            dtype: tensor.dtype,
            layout: tensor.layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() -> Result<()> {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(
            data.clone(),
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.dtype(), DataType::F32);
        assert_eq!(tensor.numel(), 4);

        let extracted = tensor.to_vec()?;
        assert_eq!(extracted, data);

        Ok(())
    }

    #[test]
    fn test_zeros_and_ones() -> Result<()> {
        let zeros = Tensor::zeros(vec![3, 3], DataType::F32, TensorLayout::RowMajor)?;
        let zeros_data = zeros.to_vec()?;
        assert!(zeros_data.iter().all(|&x| x == 0.0));

        let ones = Tensor::ones(vec![2, 3], DataType::F32, TensorLayout::RowMajor)?;
        let ones_data = ones.to_vec()?;
        assert!(ones_data.iter().all(|&x| x == 1.0));

        Ok(())
    }

    #[test]
    fn test_broadcasting() {
        // Compatible shapes
        assert_eq!(
            Tensor::broadcast_shape(&[3, 1], &[1, 4]).unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            Tensor::broadcast_shape(&[2, 3, 1], &[1, 4]).unwrap(),
            vec![2, 3, 4]
        );

        // Incompatible shapes
        assert!(Tensor::broadcast_shape(&[3, 2], &[2, 3]).is_err());
    }

    #[test]
    fn test_broadcastable_check() -> Result<()> {
        let tensor1 = Tensor::zeros(vec![3, 1], DataType::F32, TensorLayout::RowMajor)?;
        let tensor2 = Tensor::zeros(vec![1, 4], DataType::F32, TensorLayout::RowMajor)?;
        let tensor3 = Tensor::zeros(vec![2, 3], DataType::F32, TensorLayout::RowMajor)?;

        assert!(tensor1.is_broadcastable_with(&tensor2));
        assert!(!tensor1.is_broadcastable_with(&tensor3));

        Ok(())
    }

    #[test]
    fn test_data_type_conversions() -> Result<()> {
        // Test F16 conversion
        let data = vec![1.5, 2.5, 3.5, 4.5];
        let tensor_f16 = Tensor::from_data(
            data.clone(),
            vec![2, 2],
            DataType::F16,
            TensorLayout::RowMajor,
        )?;
        let extracted_f16 = tensor_f16.to_vec()?;

        // F16 has limited precision, so we check with tolerance
        for (original, extracted) in data.iter().zip(extracted_f16.iter()) {
            assert!((original - extracted).abs() < 0.01);
        }

        // Test I8 conversion
        let int_data = vec![1.0, -2.0, 3.0, -4.0];
        let tensor_i8 =
            Tensor::from_data(int_data, vec![2, 2], DataType::I8, TensorLayout::RowMajor)?;
        let extracted_i8 = tensor_i8.to_vec()?;
        assert_eq!(extracted_i8, vec![1.0, -2.0, 3.0, -4.0]);

        Ok(())
    }

    #[test]
    fn test_device_operations() -> Result<()> {
        let tensor = Tensor::zeros(vec![2, 2], DataType::F32, TensorLayout::RowMajor)?;

        // Should be on CPU by default
        assert!(matches!(tensor.device(), Device::Cpu));

        // CPU conversion should work
        let cpu_tensor = tensor.to_cpu()?;
        assert!(matches!(cpu_tensor.device(), Device::Cpu));

        Ok(())
    }
}
