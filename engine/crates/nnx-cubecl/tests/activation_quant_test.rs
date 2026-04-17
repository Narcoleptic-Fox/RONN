//! Tests for GPU activation quantization kernels (Q8_0 round-trip).

#[cfg(feature = "wgpu")]
mod wgpu_tests {
    use nnx_cubecl::activation_quant::{
        dequantize_q8_0_to_f32_kernel, quantize_f32_to_q8_0_kernel,
    };
    use nnx_cubecl::backend::CubeclBackend;
    use nnx_cubecl::WgpuRuntime;
    use nnx_core::backend::KernelBackend;

    use cubecl::prelude::*;

    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_q8_0_roundtrip_single_block() {
        let backend = CubeclBackend::<WgpuRuntime>::new();

        // Single block of 32 values
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.1).collect();
        let input = backend.from_f32(&values);

        let num_blocks = 1;
        let scales = backend.zeros(num_blocks);
        let quants = backend.from_u32(&vec![0u32; num_blocks * 8]);

        // Quantize
        unsafe {
            quantize_f32_to_q8_0_kernel::launch::<WgpuRuntime>(
                backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&input.handle, input.len, 1),
                ArrayArg::from_raw_parts::<f32>(&scales.handle, scales.len, 1),
                ArrayArg::from_raw_parts::<u32>(&quants.handle, quants.len, 1),
                ScalarArg::new(32u32),
            );
        }

        // Dequantize
        let output = backend.zeros(32);
        unsafe {
            dequantize_q8_0_to_f32_kernel::launch::<WgpuRuntime>(
                backend.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&scales.handle, scales.len, 1),
                ArrayArg::from_raw_parts::<u32>(&quants.handle, quants.len, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, output.len, 1),
                ScalarArg::new(32u32),
            );
        }

        let result = backend.to_f32(&output);
        let error = max_abs_error(&values, &result);

        // Q8_0 should have very low error (max_abs / 127, typically < 0.02 for small values)
        assert!(
            error < 0.02,
            "Q8_0 roundtrip error {} exceeds 0.02",
            error
        );
    }

    #[test]
    fn test_q8_0_roundtrip_multiple_blocks() {
        let backend = CubeclBackend::<WgpuRuntime>::new();

        // 3 blocks (96 values)
        let numel = 96;
        let values: Vec<f32> = (0..numel).map(|i| ((i % 32) as f32 - 15.5) * 0.15).collect();
        let input = backend.from_f32(&values);

        let num_blocks = 3;
        let scales = backend.zeros(num_blocks);
        let quants = backend.from_u32(&vec![0u32; num_blocks * 8]);

        // Quantize
        unsafe {
            quantize_f32_to_q8_0_kernel::launch::<WgpuRuntime>(
                backend.client(),
                CubeCount::Static(num_blocks as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&input.handle, input.len, 1),
                ArrayArg::from_raw_parts::<f32>(&scales.handle, scales.len, 1),
                ArrayArg::from_raw_parts::<u32>(&quants.handle, quants.len, 1),
                ScalarArg::new(numel as u32),
            );
        }

        // Dequantize
        let output = backend.zeros(numel);
        unsafe {
            dequantize_q8_0_to_f32_kernel::launch::<WgpuRuntime>(
                backend.client(),
                CubeCount::Static(num_blocks as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&scales.handle, scales.len, 1),
                ArrayArg::from_raw_parts::<u32>(&quants.handle, quants.len, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, output.len, 1),
                ScalarArg::new(numel as u32),
            );
        }

        let result = backend.to_f32(&output);
        let error = max_abs_error(&values, &result);

        assert!(
            error < 0.03,
            "Q8_0 multi-block roundtrip error {} exceeds 0.03",
            error
        );
    }

    #[test]
    fn test_q8_0_roundtrip_partial_block() {
        let backend = CubeclBackend::<WgpuRuntime>::new();

        // Partial block (37 values, spans 2 blocks)
        let numel = 37;
        let values: Vec<f32> = (0..numel).map(|i| (i as f32 - 18.0) * 0.08).collect();
        let input = backend.from_f32(&values);

        let num_blocks = (numel + 31) / 32;
        let scales = backend.zeros(num_blocks);
        let quants = backend.from_u32(&vec![0u32; num_blocks * 8]);

        // Quantize
        unsafe {
            quantize_f32_to_q8_0_kernel::launch::<WgpuRuntime>(
                backend.client(),
                CubeCount::Static(num_blocks as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&input.handle, input.len, 1),
                ArrayArg::from_raw_parts::<f32>(&scales.handle, scales.len, 1),
                ArrayArg::from_raw_parts::<u32>(&quants.handle, quants.len, 1),
                ScalarArg::new(numel as u32),
            );
        }

        // Dequantize
        let output = backend.zeros(numel);
        unsafe {
            dequantize_q8_0_to_f32_kernel::launch::<WgpuRuntime>(
                backend.client(),
                CubeCount::Static(num_blocks as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&scales.handle, scales.len, 1),
                ArrayArg::from_raw_parts::<u32>(&quants.handle, quants.len, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, output.len, 1),
                ScalarArg::new(numel as u32),
            );
        }

        let result = backend.to_f32(&output);
        let error = max_abs_error(&values, &result);

        assert!(
            error < 0.02,
            "Q8_0 partial-block roundtrip error {} exceeds 0.02",
            error
        );
    }

    #[test]
    fn test_q8_0_backend_helpers() {
        // Test the high-level backend helper methods
        let backend = CubeclBackend::<WgpuRuntime>::new();

        let numel = 64;
        let values: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 1.6).collect();
        let input = backend.from_f32(&values);

        let (mut scales, mut quants) = backend.alloc_q8_0_buffers(numel);

        // Use helper methods
        backend.quantize_f32_to_q8_0(&input, &mut scales, &mut quants);

        let mut output = backend.zeros(numel);
        backend.dequantize_q8_0_to_f32(&scales, &quants, &mut output);

        let result = backend.to_f32(&output);
        let error = max_abs_error(&values, &result);

        assert!(
            error < 0.03,
            "Q8_0 backend helper roundtrip error {} exceeds 0.03",
            error
        );
    }
}
