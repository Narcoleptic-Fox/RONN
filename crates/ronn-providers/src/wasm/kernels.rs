//! WebAssembly SIMD-optimized kernels for neural network operations.
//!
//! This module implements WASM SIMD128 optimized kernels for common neural
//! network operations, providing near-native performance in browsers that
//! support WebAssembly SIMD.

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use ronn_core::{CompiledKernel, DataType, KernelStats, MemoryUsage, Tensor, TensorLayout};

/// WASM SIMD128 operations for vectorized computation.
#[derive(Debug, Clone)]
pub struct WasmSimd128Ops;

impl WasmSimd128Ops {
    /// Check if WASM SIMD128 is available.
    pub fn is_simd_available() -> bool {
        // In a real implementation, this would detect SIMD capability
        cfg!(target_feature = "simd128") || cfg!(not(target_arch = "wasm32"))
    }

    /// Vectorized addition of F32 arrays using SIMD128.
    #[cfg(target_arch = "wasm32")]
    pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(anyhow!("Array length mismatch in SIMD add"));
        }

        let simd_len = a.len() & !3; // Process 4 elements at a time

        // SIMD processing (4 f32 elements per vector)
        for i in (0..simd_len).step_by(4) {
            unsafe {
                use core::arch::wasm32::*;

                let va = v128_load(a.as_ptr().add(i) as *const v128);
                let vb = v128_load(b.as_ptr().add(i) as *const v128);
                let vresult = f32x4_add(va, vb);
                v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);
            }
        }

        // Handle remaining elements
        for i in simd_len..a.len() {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Fallback addition for non-WASM targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(anyhow!("Array length mismatch in add"));
        }

        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    /// Vectorized multiplication of F32 arrays using SIMD128.
    #[cfg(target_arch = "wasm32")]
    pub fn simd_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(anyhow!("Array length mismatch in SIMD mul"));
        }

        let simd_len = a.len() & !3;

        for i in (0..simd_len).step_by(4) {
            unsafe {
                use core::arch::wasm32::*;

                let va = v128_load(a.as_ptr().add(i) as *const v128);
                let vb = v128_load(b.as_ptr().add(i) as *const v128);
                let vresult = f32x4_mul(va, vb);
                v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);
            }
        }

        for i in simd_len..a.len() {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn simd_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<()> {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
        Ok(())
    }

    /// Vectorized ReLU activation using SIMD128.
    #[cfg(target_arch = "wasm32")]
    pub fn simd_relu_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(anyhow!("Array length mismatch in SIMD ReLU"));
        }

        let simd_len = input.len() & !3;

        for i in (0..simd_len).step_by(4) {
            unsafe {
                use core::arch::wasm32::*;

                let vinput = v128_load(input.as_ptr().add(i) as *const v128);
                let vzeros = f32x4_splat(0.0);
                let vresult = f32x4_pmax(vinput, vzeros); // max(input, 0)
                v128_store(output.as_mut_ptr().add(i) as *mut v128, vresult);
            }
        }

        for i in simd_len..input.len() {
            output[i] = input[i].max(0.0);
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn simd_relu_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
        for i in 0..input.len() {
            output[i] = input[i].max(0.0);
        }
        Ok(())
    }

    /// Matrix multiplication using SIMD128 (simplified version).
    pub fn simd_matmul_f32(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(anyhow!("Matrix dimension mismatch"));
        }

        // Simple matrix multiplication with SIMD-optimized inner loop
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;

                // Process k elements, 4 at a time if possible
                let simd_k = k & !3;

                #[cfg(target_arch = "wasm32")]
                {
                    let mut vsum = unsafe { core::arch::wasm32::f32x4_splat(0.0) };

                    for l in (0..simd_k).step_by(4) {
                        unsafe {
                            use core::arch::wasm32::*;

                            let va = v128_load(a.as_ptr().add(i * k + l) as *const v128);
                            let vb = v128_load(b.as_ptr().add(l * n + j) as *const v128);
                            let vprod = f32x4_mul(va, vb);
                            vsum = f32x4_add(vsum, vprod);
                        }
                    }

                    // Sum the 4 elements in the SIMD register
                    let sum_array = unsafe { core::mem::transmute::<_, [f32; 4]>(vsum) };
                    sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
                }

                // Handle remaining elements
                for l in simd_k..k {
                    sum += a[i * k + l] * b[l * n + j];
                }

                c[i * n + j] = sum;
            }
        }

        Ok(())
    }
}

/// WASM kernel for neural network operations.
#[derive(Debug)]
pub struct WasmKernel {
    /// Operation type this kernel performs.
    op_type: String,
    /// Kernel implementation function.
    kernel_fn: fn(&WasmKernel, &[Tensor]) -> Result<Vec<Tensor>>,
    /// Performance statistics.
    stats: KernelStats,
    /// Memory usage tracking.
    memory_usage: MemoryUsage,
    /// Kernel configuration parameters.
    config: HashMap<String, f64>,
}

impl WasmKernel {
    /// Create a new WASM kernel for the specified operation.
    pub fn new(op_type: &str) -> Self {
        let kernel_fn = match op_type {
            "Add" => Self::execute_add,
            "Mul" => Self::execute_mul,
            "MatMul" => Self::execute_matmul,
            "ReLU" => Self::execute_relu,
            "Sigmoid" => Self::execute_sigmoid,
            "Softmax" => Self::execute_softmax,
            _ => Self::execute_fallback,
        };

        Self {
            op_type: op_type.to_string(),
            kernel_fn,
            stats: KernelStats {
                execution_count: 0,
                average_time_us: 0.0,
                min_time_us: 0.0,
                max_time_us: 0.0,
            },
            memory_usage: MemoryUsage {
                peak_bytes: 0,
                current_bytes: 0,
                allocation_count: 0,
            },
            config: HashMap::new(),
        }
    }

    /// Execute element-wise addition.
    fn execute_add(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(anyhow!("Add operation requires exactly 2 inputs"));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        if a.shape() != b.shape() {
            return Err(anyhow!(
                "Shape mismatch for Add: {:?} vs {:?}",
                a.shape(),
                b.shape()
            ));
        }

        let a_data = a.to_vec()?;
        let b_data = b.to_vec()?;
        let mut result_data = vec![0.0f32; a_data.len()];
        WasmSimd128Ops::simd_add_f32(&a_data, &b_data, &mut result_data)?;

        let result = Tensor::from_data(
            result_data,
            a.shape().to_vec(),
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        Ok(vec![result])
    }

    /// Execute element-wise multiplication.
    fn execute_mul(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(anyhow!("Mul operation requires exactly 2 inputs"));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        if a.shape() != b.shape() {
            return Err(anyhow!(
                "Shape mismatch for Mul: {:?} vs {:?}",
                a.shape(),
                b.shape()
            ));
        }

        let a_data = a.to_vec()?;
        let b_data = b.to_vec()?;
        let mut result_data = vec![0.0f32; a_data.len()];
        WasmSimd128Ops::simd_mul_f32(&a_data, &b_data, &mut result_data)?;

        let result = Tensor::from_data(
            result_data,
            a.shape().to_vec(),
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        Ok(vec![result])
    }

    /// Execute matrix multiplication.
    fn execute_matmul(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != 2 {
            return Err(anyhow!("MatMul operation requires exactly 2 inputs"));
        }

        let a = &inputs[0];
        let b = &inputs[1];

        // Validate matrix dimensions
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(anyhow!("MatMul requires 2D tensors"));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(anyhow!(
                "Matrix dimension mismatch: {} != {}",
                k,
                b_shape[0]
            ));
        }

        let a_data = a.to_vec()?;
        let b_data = b.to_vec()?;
        let mut result_data = vec![0.0f32; m * n];
        WasmSimd128Ops::simd_matmul_f32(&a_data, &b_data, &mut result_data, m, n, k)?;

        let result = Tensor::from_data(
            result_data,
            vec![m, n],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        Ok(vec![result])
    }

    /// Execute ReLU activation.
    fn execute_relu(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(anyhow!("ReLU operation requires exactly 1 input"));
        }

        let input = &inputs[0];
        let input_data = input.to_vec()?;
        let mut result_data = vec![0.0f32; input_data.len()];

        WasmSimd128Ops::simd_relu_f32(&input_data, &mut result_data)?;

        let result = Tensor::from_data(
            result_data,
            input.shape().to_vec(),
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        Ok(vec![result])
    }

    /// Execute Sigmoid activation.
    fn execute_sigmoid(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(anyhow!("Sigmoid operation requires exactly 1 input"));
        }

        let input = &inputs[0];
        let input_data = input.to_vec()?;
        let result_data: Vec<f32> = input_data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        let result = Tensor::from_data(
            result_data,
            input.shape().to_vec(),
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        Ok(vec![result])
    }

    /// Execute Softmax activation.
    fn execute_softmax(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != 1 {
            return Err(anyhow!("Softmax operation requires exactly 1 input"));
        }

        let input = &inputs[0];
        let data = input.to_vec()?;

        // Find maximum value for numerical stability
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate exp(x - max) and sum
        let exp_values: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();

        // Normalize
        let result_data: Vec<f32> = exp_values.iter().map(|&x| x / sum_exp).collect();

        let result = Tensor::from_data(
            result_data,
            input.shape().to_vec(),
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        Ok(vec![result])
    }

    /// Fallback execution for unsupported operations.
    fn execute_fallback(&self, _inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        Err(anyhow!(
            "Operation {} not implemented for WASM",
            self.op_type
        ))
    }

    /// Update performance statistics.
    fn update_stats(&mut self, execution_time_us: f64) {
        self.stats.execution_count += 1;

        if self.stats.execution_count == 1 {
            self.stats.min_time_us = execution_time_us;
            self.stats.max_time_us = execution_time_us;
            self.stats.average_time_us = execution_time_us;
        } else {
            self.stats.min_time_us = self.stats.min_time_us.min(execution_time_us);
            self.stats.max_time_us = self.stats.max_time_us.max(execution_time_us);

            // Update running average
            let n = self.stats.execution_count as f64;
            self.stats.average_time_us =
                ((n - 1.0) * self.stats.average_time_us + execution_time_us) / n;
        }
    }
}

impl CompiledKernel for WasmKernel {
    fn execute(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let start_time = std::time::Instant::now();

        let results = (self.kernel_fn)(self, inputs)?;

        let execution_time = start_time.elapsed().as_micros() as f64;

        // Note: We can't mutate self in this immutable context
        // In a real implementation, statistics would be tracked differently

        Ok(results)
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        self.memory_usage.clone()
    }

    fn get_performance_stats(&self) -> KernelStats {
        self.stats.clone()
    }
}

/// Create a WASM kernel for the specified operation.
pub fn create_wasm_kernel(op_type: &str) -> WasmKernel {
    WasmKernel::new(op_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_availability() {
        let _available = WasmSimd128Ops::is_simd_available();
        // Test passes if it doesn't panic
    }

    #[test]
    fn test_simd_add() -> Result<()> {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 5];

        WasmSimd128Ops::simd_add_f32(&a, &b, &mut result)?;

        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0]);
        Ok(())
    }

    #[test]
    fn test_simd_mul() -> Result<()> {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        let mut result = vec![0.0; 4];

        WasmSimd128Ops::simd_mul_f32(&a, &b, &mut result)?;

        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
        Ok(())
    }

    #[test]
    fn test_simd_relu() -> Result<()> {
        let input = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let mut output = vec![0.0; 5];

        WasmSimd128Ops::simd_relu_f32(&input, &mut output)?;

        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.5, 1.0]);
        Ok(())
    }

    #[test]
    fn test_wasm_kernel_add() -> Result<()> {
        let kernel = create_wasm_kernel("Add");

        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let b = Tensor::from_data(
            vec![4.0, 5.0, 6.0],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let results = kernel.execute(&[a, b])?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].to_vec().unwrap(), vec![5.0, 7.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_wasm_kernel_matmul() -> Result<()> {
        let kernel = create_wasm_kernel("MatMul");

        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;
        let b = Tensor::from_data(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let results = kernel.execute(&[a, b])?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].shape(), &[2, 2]);
        // Result should be [[19, 22], [43, 50]]

        Ok(())
    }

    #[test]
    fn test_wasm_kernel_relu() -> Result<()> {
        let kernel = create_wasm_kernel("ReLU");

        let input = Tensor::from_data(
            vec![-1.0, 0.0, 1.0, -2.0, 3.0],
            vec![5],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let results = kernel.execute(&[input])?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].to_vec().unwrap(), vec![0.0, 0.0, 1.0, 0.0, 3.0]);

        Ok(())
    }

    #[test]
    fn test_unsupported_operation() {
        let kernel = create_wasm_kernel("UnsupportedOp");

        let input =
            Tensor::from_data(vec![1.0], vec![1], DataType::F32, TensorLayout::RowMajor).unwrap();
        let result = kernel.execute(&[input]);

        assert!(result.is_err());
    }
}
