//! Sparse execution provider trait extension for activation sparsity.
//!
//! Extends RONN's execution provider framework to support partial-layer
//! execution where only predicted-active neurons are computed.

use ronn_core::tensor::Tensor;
use ronn_core::types::ProviderId;

/// Trait for execution providers that support sparse (partial-layer) execution.
///
/// A single FFN layer may execute across both GPU (hot neurons) and CPU
/// (cold neurons) simultaneously. Providers implementing this trait can
/// handle the partial computation.
pub trait SparseExecutionProvider: Send + Sync {
    /// Get the provider ID.
    fn provider_id(&self) -> ProviderId;

    /// Execute a sparse linear operation: only compute output rows at `active_indices`.
    ///
    /// # Arguments
    /// * `input` - Input tensor, shape [batch_size, input_dim] or [input_dim]
    /// * `weight` - Weight matrix, shape [output_dim, input_dim]
    /// * `bias` - Optional bias vector, shape [output_dim]
    /// * `active_indices` - Indices of output neurons to compute
    /// * `output_dim` - Total output dimension
    ///
    /// # Returns
    /// Sparse output tensor with only active indices populated.
    fn execute_sparse(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        active_indices: &[usize],
        output_dim: usize,
    ) -> anyhow::Result<Tensor>;

    /// Check if this provider supports sparse execution.
    fn supports_sparse(&self) -> bool {
        true
    }

    /// Estimate the time (microseconds) to compute `num_active` neurons
    /// out of `total` neurons for the given input dimension.
    fn estimate_sparse_time_us(
        &self,
        input_dim: usize,
        num_active: usize,
        total: usize,
    ) -> f64 {
        // Default: linear estimate based on active ratio
        let active_ratio = num_active as f64 / total.max(1) as f64;
        let base_time = (input_dim * total) as f64 * 0.001; // rough ns per multiply
        base_time * active_ratio
    }
}

/// CPU sparse execution using gather-scatter or masked approaches.
pub struct CpuSparseProvider;

impl SparseExecutionProvider for CpuSparseProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::CPU
    }

    fn execute_sparse(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        active_indices: &[usize],
        output_dim: usize,
    ) -> anyhow::Result<Tensor> {
        use ronn_core::types::{DataType, TensorLayout};

        let input_data = input.to_vec()?;
        let weight_data = weight.to_vec()?;
        let bias_data = bias.map(|b| b.to_vec()).transpose()?;

        let input_shape = input.shape();
        let input_dim = *input_shape.last().unwrap_or(&0);
        let batch_size = if input_shape.len() == 1 {
            1
        } else {
            input_data.len() / input_dim
        };

        let mut output_data = vec![0.0f32; batch_size * output_dim];

        for &out_idx in active_indices {
            if out_idx >= output_dim {
                continue;
            }
            let weight_row_offset = out_idx * input_dim;

            for batch in 0..batch_size {
                let input_offset = batch * input_dim;
                let output_offset = batch * output_dim + out_idx;

                let mut sum = bias_data
                    .as_ref()
                    .and_then(|b| b.get(out_idx).copied())
                    .unwrap_or(0.0);

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

        Ok(Tensor::from_data(output_data, out_shape, DataType::F32, TensorLayout::RowMajor)?)
    }
}

/// GPU sparse execution (delegates to standard GPU provider for now,
/// with sparse subset selection happening on CPU before GPU kernel launch).
pub struct GpuSparseProvider;

impl SparseExecutionProvider for GpuSparseProvider {
    fn provider_id(&self) -> ProviderId {
        ProviderId::GPU
    }

    fn execute_sparse(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        active_indices: &[usize],
        output_dim: usize,
    ) -> anyhow::Result<Tensor> {
        // For now, delegate to CPU implementation.
        // Full GPU implementation would use compute shaders or cudarc
        // to execute sparse kernels directly on GPU.
        let cpu = CpuSparseProvider;
        cpu.execute_sparse(input, weight, bias, active_indices, output_dim)
    }

    fn estimate_sparse_time_us(
        &self,
        input_dim: usize,
        num_active: usize,
        total: usize,
    ) -> f64 {
        // GPU is faster for larger workloads
        let active_ratio = num_active as f64 / total.max(1) as f64;
        let base_time = (input_dim * total) as f64 * 0.0001; // GPU is ~10x faster
        base_time * active_ratio + 10.0 // kernel launch overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::tensor::Tensor;
    use ronn_core::types::{DataType, TensorLayout};

    #[test]
    fn test_cpu_sparse_provider() -> anyhow::Result<()> {
        let provider = CpuSparseProvider;

        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let weight = Tensor::from_data(
            vec![
                1.0, 0.0, 0.0, // neuron 0: picks input[0]
                0.0, 1.0, 0.0, // neuron 1: picks input[1]
                0.0, 0.0, 1.0, // neuron 2: picks input[2]
                1.0, 1.0, 1.0, // neuron 3: sum of all
            ],
            vec![4, 3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Only compute neurons 0 and 3
        let result = provider.execute_sparse(&input, &weight, None, &[0, 3], 4)?;
        let data = result.to_vec()?;

        assert!((data[0] - 1.0).abs() < 1e-5); // neuron 0
        assert!((data[1]).abs() < 1e-5); // neuron 1 (skipped)
        assert!((data[2]).abs() < 1e-5); // neuron 2 (skipped)
        assert!((data[3] - 6.0).abs() < 1e-5); // neuron 3: 1+2+3

        Ok(())
    }

    #[test]
    fn test_sparse_provider_supports() {
        let cpu = CpuSparseProvider;
        assert!(cpu.supports_sparse());
        assert_eq!(cpu.provider_id(), ProviderId::CPU);

        let gpu = GpuSparseProvider;
        assert!(gpu.supports_sparse());
        assert_eq!(gpu.provider_id(), ProviderId::GPU);
    }
}
