//! Sparse matrix operations for selective neuron computation.
//!
//! Implements two approaches for computing only the predicted-active neurons:
//! 1. **Gather-Scatter**: Extract active weight rows -> dense matmul -> scatter results
//! 2. **Masked Sparse**: Zero-mask inactive neurons in the weight matrix before multiply
//!
//! Both approaches skip computation for inactive neurons, providing speedup
//! proportional to the sparsity ratio.

use crate::error::{Result, SparsityError};
use ronn_core::tensor::Tensor;
use ronn_core::types::{DataType, TensorLayout};
use tracing::debug;

/// Strategy for sparse matrix multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseStrategy {
    /// Gather active rows, dense matmul, scatter results.
    /// Best for high sparsity (>70% inactive).
    GatherScatter,
    /// Apply mask to weight matrix, then standard matmul.
    /// Best for moderate sparsity (30-70% inactive).
    MaskedDense,
    /// Automatically select based on sparsity ratio.
    Auto,
}

/// Sparse matrix multiply: compute `input @ weight^T + bias` but only for
/// the neuron indices specified in `active_indices`.
///
/// # Arguments
/// * `input` - Input hidden state, shape [batch_size, input_dim] or [input_dim]
/// * `weight` - Full weight matrix, shape [output_dim, input_dim]
/// * `bias` - Optional bias vector, shape [output_dim]
/// * `active_indices` - Indices of neurons (output rows) to compute
/// * `output_dim` - Total output dimension (for result placement)
///
/// # Returns
/// Output tensor of shape [batch_size, output_dim] or [output_dim] with
/// only the active indices populated, rest zero.
pub fn sparse_linear(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    active_indices: &[usize],
    output_dim: usize,
) -> Result<Tensor> {
    let input_data = input
        .to_vec()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read input: {}", e)))?;
    let weight_data = weight
        .to_vec()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read weight: {}", e)))?;
    let bias_data = bias
        .map(|b| b.to_vec())
        .transpose()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read bias: {}", e)))?;

    let input_shape = input.shape();
    let weight_shape = weight.shape();

    let input_dim = *input_shape.last().ok_or_else(|| {
        SparsityError::SparseOp("Input tensor is scalar".into())
    })?;

    if weight_shape.len() != 2 || weight_shape[1] != input_dim {
        return Err(SparsityError::SparseOp(format!(
            "Weight shape {:?} incompatible with input dim {}",
            weight_shape, input_dim
        )));
    }

    // Determine batch size
    let batch_size = if input_shape.len() == 1 {
        1
    } else {
        input_data.len() / input_dim
    };

    let mut output_data = vec![0.0f32; batch_size * output_dim];

    // Only compute the active rows of the output
    for &out_idx in active_indices {
        if out_idx >= output_dim {
            continue;
        }
        let weight_row_offset = out_idx * input_dim;

        for batch in 0..batch_size {
            let input_offset = batch * input_dim;
            let output_offset = batch * output_dim + out_idx;

            let mut sum = if let Some(ref bd) = bias_data {
                if out_idx < bd.len() {
                    bd[out_idx]
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Dot product of input row with weight row
            for j in 0..input_dim {
                sum += input_data[input_offset + j] * weight_data[weight_row_offset + j];
            }

            output_data[output_offset] = sum;
        }
    }

    let out_shape = if batch_size == 1 {
        vec![output_dim]
    } else {
        vec![batch_size, output_dim]
    };

    Tensor::from_data(output_data, out_shape, DataType::F32, TensorLayout::RowMajor)
        .map_err(|e| SparsityError::SparseOp(format!("Failed to create output tensor: {}", e)))
}

/// Gather-scatter approach: extract active weight rows, perform dense matmul
/// on the smaller matrix, then scatter results back.
///
/// More efficient than sparse_linear when sparsity is very high because the
/// dense matmul on the gathered subset can leverage SIMD/BLAS optimizations.
pub fn gather_scatter_linear(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    active_indices: &[usize],
    output_dim: usize,
) -> Result<Tensor> {
    if active_indices.is_empty() {
        let shape = if input.shape().len() == 1 {
            vec![output_dim]
        } else {
            vec![input.shape()[0], output_dim]
        };
        return Tensor::zeros(shape, DataType::F32, TensorLayout::RowMajor)
            .map_err(|e| SparsityError::SparseOp(format!("Failed to create zero tensor: {}", e)));
    }

    let input_data = input
        .to_vec()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read input: {}", e)))?;
    let weight_data = weight
        .to_vec()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read weight: {}", e)))?;
    let bias_data = bias
        .map(|b| b.to_vec())
        .transpose()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read bias: {}", e)))?;

    let input_shape = input.shape();
    let input_dim = *input_shape.last().unwrap();
    let batch_size = if input_shape.len() == 1 {
        1
    } else {
        input_data.len() / input_dim
    };
    let active_count = active_indices.len();

    // GATHER: extract only the active weight rows
    let mut gathered_weights = vec![0.0f32; active_count * input_dim];
    let mut gathered_bias = vec![0.0f32; active_count];
    for (i, &out_idx) in active_indices.iter().enumerate() {
        let src_offset = out_idx * input_dim;
        let dst_offset = i * input_dim;
        gathered_weights[dst_offset..dst_offset + input_dim]
            .copy_from_slice(&weight_data[src_offset..src_offset + input_dim]);

        if let Some(ref bd) = bias_data {
            if out_idx < bd.len() {
                gathered_bias[i] = bd[out_idx];
            }
        }
    }

    // DENSE MATMUL on the smaller gathered matrix: [batch, input_dim] @ [active_count, input_dim]^T
    let mut gathered_output = vec![0.0f32; batch_size * active_count];
    for batch in 0..batch_size {
        let input_offset = batch * input_dim;
        for (i, _) in active_indices.iter().enumerate() {
            let w_offset = i * input_dim;
            let mut sum = gathered_bias[i];
            for j in 0..input_dim {
                sum += input_data[input_offset + j] * gathered_weights[w_offset + j];
            }
            gathered_output[batch * active_count + i] = sum;
        }
    }

    // SCATTER: place gathered results into full output
    let mut output_data = vec![0.0f32; batch_size * output_dim];
    for batch in 0..batch_size {
        for (i, &out_idx) in active_indices.iter().enumerate() {
            if out_idx < output_dim {
                output_data[batch * output_dim + out_idx] =
                    gathered_output[batch * active_count + i];
            }
        }
    }

    let out_shape = if batch_size == 1 {
        vec![output_dim]
    } else {
        vec![batch_size, output_dim]
    };

    Tensor::from_data(output_data, out_shape, DataType::F32, TensorLayout::RowMajor)
        .map_err(|e| SparsityError::SparseOp(format!("Failed to create output tensor: {}", e)))
}

/// Automatically select and execute the best sparse strategy based on
/// the sparsity ratio.
pub fn sparse_linear_auto(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    active_indices: &[usize],
    output_dim: usize,
    strategy: SparseStrategy,
) -> Result<Tensor> {
    let sparsity_ratio = 1.0 - (active_indices.len() as f64 / output_dim.max(1) as f64);

    let effective_strategy = match strategy {
        SparseStrategy::Auto => {
            if sparsity_ratio > 0.7 {
                SparseStrategy::GatherScatter
            } else {
                SparseStrategy::MaskedDense
            }
        }
        other => other,
    };

    debug!(
        "Sparse linear: {} active / {} total ({:.1}% sparse), strategy: {:?}",
        active_indices.len(),
        output_dim,
        sparsity_ratio * 100.0,
        effective_strategy
    );

    match effective_strategy {
        SparseStrategy::GatherScatter => {
            gather_scatter_linear(input, weight, bias, active_indices, output_dim)
        }
        SparseStrategy::MaskedDense | SparseStrategy::Auto => {
            sparse_linear(input, weight, bias, active_indices, output_dim)
        }
    }
}

/// Apply an activation function to a tensor, optionally only at specified indices.
pub fn sparse_activation(
    input: &Tensor,
    active_indices: Option<&[usize]>,
    activation: SparseActivation,
) -> Result<Tensor> {
    let mut data = input
        .to_vec()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read tensor: {}", e)))?;

    let indices: Vec<usize> = match active_indices {
        Some(idx) => idx.to_vec(),
        None => (0..data.len()).collect(),
    };

    for &i in &indices {
        if i < data.len() {
            data[i] = match activation {
                SparseActivation::ReLU => data[i].max(0.0),
                SparseActivation::SiLU => data[i] * (1.0 / (1.0 + (-data[i]).exp())),
                SparseActivation::GELU => {
                    let x = data[i];
                    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
                }
            };
        }
    }

    Tensor::from_data(data, input.shape(), DataType::F32, TensorLayout::RowMajor)
        .map_err(|e| SparsityError::SparseOp(format!("Failed to create output: {}", e)))
}

/// Supported activation functions for sparse computation.
#[derive(Debug, Clone, Copy)]
pub enum SparseActivation {
    ReLU,
    SiLU,
    GELU,
}

/// Combine results from hot neurons (GPU) and cold neurons (CPU) into a single
/// output tensor. Used by the scheduler when computation is split across devices.
pub fn merge_sparse_results(
    hot_result: &Tensor,
    cold_result: &Tensor,
    hot_indices: &[usize],
    cold_indices: &[usize],
    output_dim: usize,
) -> Result<Tensor> {
    let hot_data = hot_result
        .to_vec()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read hot result: {}", e)))?;
    let cold_data = cold_result
        .to_vec()
        .map_err(|e| SparsityError::SparseOp(format!("Failed to read cold result: {}", e)))?;

    let hot_shape = hot_result.shape();
    let batch_size = if hot_shape.len() == 1 {
        1
    } else {
        hot_shape[0]
    };

    let mut output = vec![0.0f32; batch_size * output_dim];

    // Place hot neuron results
    for batch in 0..batch_size {
        for (i, &idx) in hot_indices.iter().enumerate() {
            if idx < output_dim {
                let src_offset = if hot_shape.len() == 1 { i } else { batch * hot_indices.len() + i };
                if src_offset < hot_data.len() {
                    output[batch * output_dim + idx] = hot_data[src_offset];
                }
            }
        }
        // Place cold neuron results
        for (i, &idx) in cold_indices.iter().enumerate() {
            if idx < output_dim {
                let src_offset = if cold_result.shape().len() == 1 { i } else { batch * cold_indices.len() + i };
                if src_offset < cold_data.len() {
                    output[batch * output_dim + idx] = cold_data[src_offset];
                }
            }
        }
    }

    let out_shape = if batch_size == 1 {
        vec![output_dim]
    } else {
        vec![batch_size, output_dim]
    };

    Tensor::from_data(output, out_shape, DataType::F32, TensorLayout::RowMajor)
        .map_err(|e| SparsityError::SparseOp(format!("Failed to create merged output: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "index {}: {} vs {} (diff {})",
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    /// Dense linear for reference comparison.
    fn dense_linear(input: &[f32], weight: &[f32], bias: Option<&[f32]>, input_dim: usize, output_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; output_dim];
        for i in 0..output_dim {
            let mut sum = bias.map_or(0.0, |b| b[i]);
            for j in 0..input_dim {
                sum += input[j] * weight[i * input_dim + j];
            }
            output[i] = sum;
        }
        output
    }

    #[test]
    fn test_sparse_linear_correctness() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let input_dim = 4;
        let output_dim = 8;
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let weight_data: Vec<f32> = (0..output_dim * input_dim).map(|i| (i as f32) * 0.1).collect();
        let bias_data = vec![0.1; output_dim];

        let input = Tensor::from_data(input_data.clone(), vec![input_dim], DataType::F32, TensorLayout::RowMajor)?;
        let weight = Tensor::from_data(weight_data.clone(), vec![output_dim, input_dim], DataType::F32, TensorLayout::RowMajor)?;
        let bias = Tensor::from_data(bias_data.clone(), vec![output_dim], DataType::F32, TensorLayout::RowMajor)?;

        // Compute dense reference
        let dense_result = dense_linear(&input_data, &weight_data, Some(&bias_data), input_dim, output_dim);

        // Compute sparse with all indices (should match dense)
        let all_indices: Vec<usize> = (0..output_dim).collect();
        let sparse_result = sparse_linear(&input, &weight, Some(&bias), &all_indices, output_dim)?;
        assert_vec_close(&sparse_result.to_vec()?, &dense_result, 1e-5);

        // Compute sparse with subset
        let subset = vec![0, 3, 7];
        let sparse_subset = sparse_linear(&input, &weight, Some(&bias), &subset, output_dim)?;
        let sparse_data = sparse_subset.to_vec()?;

        // Active indices should match dense, others should be zero
        for i in 0..output_dim {
            if subset.contains(&i) {
                assert!((sparse_data[i] - dense_result[i]).abs() < 1e-5);
            } else {
                assert!((sparse_data[i]).abs() < 1e-5);
            }
        }

        Ok(())
    }

    #[test]
    fn test_gather_scatter_correctness() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let input_dim = 4;
        let output_dim = 8;
        let input_data = vec![1.0, 0.5, -1.0, 2.0];
        let weight_data: Vec<f32> = (0..output_dim * input_dim).map(|i| ((i as f32) - 16.0) * 0.05).collect();
        let bias_data = vec![0.01; output_dim];

        let input = Tensor::from_data(input_data.clone(), vec![input_dim], DataType::F32, TensorLayout::RowMajor)?;
        let weight = Tensor::from_data(weight_data.clone(), vec![output_dim, input_dim], DataType::F32, TensorLayout::RowMajor)?;
        let bias = Tensor::from_data(bias_data.clone(), vec![output_dim], DataType::F32, TensorLayout::RowMajor)?;

        let dense_result = dense_linear(&input_data, &weight_data, Some(&bias_data), input_dim, output_dim);

        let subset = vec![1, 4, 6];
        let gs_result = gather_scatter_linear(&input, &weight, Some(&bias), &subset, output_dim)?;
        let gs_data = gs_result.to_vec()?;

        for i in 0..output_dim {
            if subset.contains(&i) {
                assert!((gs_data[i] - dense_result[i]).abs() < 1e-5,
                    "Mismatch at {}: {} vs {}", i, gs_data[i], dense_result[i]);
            } else {
                assert!((gs_data[i]).abs() < 1e-5);
            }
        }

        Ok(())
    }

    #[test]
    fn test_sparse_activation_relu() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let data = vec![-1.0, 0.5, -0.3, 2.0];
        let input = Tensor::from_data(data, vec![4], DataType::F32, TensorLayout::RowMajor)?;

        let result = sparse_activation(&input, None, SparseActivation::ReLU)?;
        let out = result.to_vec()?;
        assert_vec_close(&out, &[0.0, 0.5, 0.0, 2.0], 1e-6);

        Ok(())
    }

    #[test]
    fn test_merge_sparse_results() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let hot_data = vec![1.0, 2.0, 3.0];
        let cold_data = vec![4.0, 5.0];
        let hot = Tensor::from_data(hot_data, vec![3], DataType::F32, TensorLayout::RowMajor)?;
        let cold = Tensor::from_data(cold_data, vec![2], DataType::F32, TensorLayout::RowMajor)?;

        let hot_indices = vec![0, 2, 4];
        let cold_indices = vec![1, 3];
        let merged = merge_sparse_results(&hot, &cold, &hot_indices, &cold_indices, 5)?;
        let out = merged.to_vec()?;

        assert_vec_close(&out, &[1.0, 4.0, 2.0, 5.0, 3.0], 1e-6);

        Ok(())
    }

    #[test]
    fn test_empty_active_indices() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let input = Tensor::from_data(vec![1.0, 2.0], vec![2], DataType::F32, TensorLayout::RowMajor)?;
        let weight = Tensor::from_data(vec![0.5; 8], vec![4, 2], DataType::F32, TensorLayout::RowMajor)?;

        let result = gather_scatter_linear(&input, &weight, None, &[], 4)?;
        let out = result.to_vec()?;
        assert!(out.iter().all(|&v| v == 0.0));

        Ok(())
    }

    #[test]
    fn test_batch_sparse_linear() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let input_dim = 3;
        let output_dim = 4;
        // Batch of 2
        let input_data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let weight_data = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let input = Tensor::from_data(input_data, vec![2, input_dim], DataType::F32, TensorLayout::RowMajor)?;
        let weight = Tensor::from_data(weight_data, vec![output_dim, input_dim], DataType::F32, TensorLayout::RowMajor)?;

        let active = vec![0, 2, 3];
        let result = sparse_linear(&input, &weight, None, &active, output_dim)?;
        let out = result.to_vec()?;

        // Batch 0: input=[1,0,1] => neuron0=1, neuron2=1, neuron3=2
        // Batch 1: input=[0,1,0] => neuron0=0, neuron2=0, neuron3=1
        assert!((out[0] - 1.0).abs() < 1e-5); // batch0, neuron0
        assert!((out[1] - 0.0).abs() < 1e-5); // batch0, neuron1 (inactive)
        assert!((out[2] - 1.0).abs() < 1e-5); // batch0, neuron2
        assert!((out[3] - 2.0).abs() < 1e-5); // batch0, neuron3
        assert!((out[4] - 0.0).abs() < 1e-5); // batch1, neuron0
        assert!((out[5] - 0.0).abs() < 1e-5); // batch1, neuron1 (inactive)
        assert!((out[6] - 0.0).abs() < 1e-5); // batch1, neuron2
        assert!((out[7] - 1.0).abs() < 1e-5); // batch1, neuron3

        Ok(())
    }
}
