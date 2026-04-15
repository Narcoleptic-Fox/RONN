//! GPU kernel correctness tests.
//!
//! Each test runs a kernel on the GPU (via wgpu) and compares the result
//! against a CPU reference implementation. Requires the `wgpu` feature.
//!
//! Run with: `cargo test -p nnx-cubecl --features wgpu`

#![cfg(feature = "wgpu")]

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

type R = WgpuRuntime;
type Server = <R as Runtime>::Server;

fn get_client() -> ComputeClient<Server> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

fn upload(client: &ComputeClient<Server>, data: &[f32]) -> cubecl::server::Handle {
    client.create(f32::as_bytes(data))
}

fn download(client: &ComputeClient<Server>, handle: &cubecl::server::Handle) -> Vec<f32> {
    let bytes = client.read_one(handle.clone());
    f32::from_bytes(&bytes).to_vec()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", msg, a.len(), b.len());
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "{}: mismatch at [{}]: gpu={}, cpu={}, diff={}",
            msg, i, av, bv, (av - bv).abs(),
        );
    }
}

// -----------------------------------------------------------------------
// CPU reference implementations
// -----------------------------------------------------------------------

fn cpu_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(v, w)| (v / rms) * w).collect()
}

fn cpu_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

fn cpu_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

fn cpu_gelu(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| {
            let cdf = 0.5 * (1.0 + (0.7978845608 * (v + 0.044715 * v * v * v)).tanh());
            v * cdf
        })
        .collect()
}

fn cpu_matvec(matrix: &[f32], vector: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0; m];
    for row in 0..m {
        let mut sum = 0.0;
        for col in 0..k {
            sum += matrix[row * k + col] * vector[col];
        }
        out[row] = sum;
    }
    out
}

fn cpu_rope(data: &mut [f32], head_dim: usize, position: usize, freq_base: f32) {
    let half = head_dim / 2;
    for pair in 0..half {
        let dim_frac = (2 * pair) as f32 / head_dim as f32;
        let freq = 1.0 / freq_base.powf(dim_frac);
        let angle = position as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let x0 = data[2 * pair];
        let x1 = data[2 * pair + 1];
        data[2 * pair] = x0 * cos - x1 * sin;
        data[2 * pair + 1] = x0 * sin + x1 * cos;
    }
}

// -----------------------------------------------------------------------
// GPU tests
// -----------------------------------------------------------------------

#[test]
fn test_silu() {
    let client = get_client();
    let input = vec![0.0, 1.0, -1.0, 2.0, -0.5, 0.5, 3.0, -3.0];
    let n = input.len();
    let expected = cpu_silu(&input);

    let input_gpu = upload(&client, &input);
    let output_gpu = upload(&client, &vec![0.0; n]);

    unsafe {
        nnx_cubecl::activations::silu_kernel::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(n as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_gpu, n, 1),
            ArrayArg::from_raw_parts::<f32>(&output_gpu, n, 1),
            ScalarArg::new(n as u32),
        );
    }

    let result = download(&client, &output_gpu);
    assert_close(&result, &expected, 1e-5, "silu");
}

#[test]
fn test_gelu() {
    let client = get_client();
    let input = vec![0.0, 1.0, -1.0, 2.0, -0.5, 0.5, 3.0, -3.0];
    let n = input.len();
    let expected = cpu_gelu(&input);

    let input_gpu = upload(&client, &input);
    let output_gpu = upload(&client, &vec![0.0; n]);

    unsafe {
        nnx_cubecl::activations::gelu_kernel::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(n as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_gpu, n, 1),
            ArrayArg::from_raw_parts::<f32>(&output_gpu, n, 1),
            ScalarArg::new(n as u32),
        );
    }

    let result = download(&client, &output_gpu);
    assert_close(&result, &expected, 1e-5, "gelu");
}

#[test]
fn test_add_inplace() {
    let client = get_client();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, 1.5, 2.5, 3.5];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let a_gpu = upload(&client, &a);
    let b_gpu = upload(&client, &b);

    unsafe {
        nnx_cubecl::activations::add_inplace_kernel::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(4, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&a_gpu, 4, 1),
            ArrayArg::from_raw_parts::<f32>(&b_gpu, 4, 1),
            ScalarArg::new(4u32),
        );
    }

    let result = download(&client, &a_gpu);
    assert_close(&result, &expected, 1e-6, "add_inplace");
}

#[test]
fn test_fused_swiglu() {
    let client = get_client();
    let gate = vec![0.5, 1.0, -1.0, 2.0];
    let up = vec![1.0, 2.0, 0.5, 3.0];
    let n = gate.len();

    let gate_silu = cpu_silu(&gate);
    let expected: Vec<f32> = gate_silu.iter().zip(up.iter()).map(|(g, u)| g * u).collect();

    let gate_gpu = upload(&client, &gate);
    let up_gpu = upload(&client, &up);
    let out_gpu = upload(&client, &vec![0.0; n]);

    unsafe {
        nnx_cubecl::activations::fused_swiglu_kernel::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(n as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&gate_gpu, n, 1),
            ArrayArg::from_raw_parts::<f32>(&up_gpu, n, 1),
            ArrayArg::from_raw_parts::<f32>(&out_gpu, n, 1),
            ScalarArg::new(n as u32),
        );
    }

    let result = download(&client, &out_gpu);
    assert_close(&result, &expected, 1e-5, "fused_swiglu");
}

#[test]
fn test_rms_norm() {
    let client = get_client();
    let input = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.0, 2.5];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let n = input.len();
    let eps = 1e-5f32;
    let expected = cpu_rms_norm(&input, &weight, eps);

    let input_gpu = upload(&client, &input);
    let weight_gpu = upload(&client, &weight);
    let output_gpu = upload(&client, &vec![0.0; n]);

    unsafe {
        nnx_cubecl::normalization::rms_norm_kernel::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&input_gpu, n, 1),
            ArrayArg::from_raw_parts::<f32>(&weight_gpu, n, 1),
            ArrayArg::from_raw_parts::<f32>(&output_gpu, n, 1),
            ScalarArg::new(n as u32),
            ScalarArg::new(eps),
        );
    }

    let result = download(&client, &output_gpu);
    assert_close(&result, &expected, 1e-4, "rms_norm");
}

#[test]
fn test_matvec() {
    let client = get_client();
    let matrix = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let vector = vec![1.0, 0.5, -1.0, 2.0];
    let m = 3;
    let k = 4;
    let expected = cpu_matvec(&matrix, &vector, m, k);

    let mat_gpu = upload(&client, &matrix);
    let vec_gpu = upload(&client, &vector);
    let out_gpu = upload(&client, &vec![0.0; m]);

    unsafe {
        nnx_cubecl::matmul::matvec_kernel::launch::<R>(
            &client,
            CubeCount::Static(m as u32, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&mat_gpu, m * k, 1),
            ArrayArg::from_raw_parts::<f32>(&vec_gpu, k, 1),
            ArrayArg::from_raw_parts::<f32>(&out_gpu, m, 1),
            ScalarArg::new(k as u32),
        );
    }

    let result = download(&client, &out_gpu);
    assert_close(&result, &expected, 1e-4, "matvec");
}

#[test]
fn test_softmax() {
    let client = get_client();
    let input = vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.5, -1.0, 3.0];
    let n = input.len();
    let expected = cpu_softmax(&input);

    let data_gpu = upload(&client, &input);

    unsafe {
        nnx_cubecl::softmax::softmax_kernel::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&data_gpu, n, 1),
            ScalarArg::new(0u32),
            ScalarArg::new(n as u32),
        );
    }

    let result = download(&client, &data_gpu);
    assert_close(&result, &expected, 1e-5, "softmax");

    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum={}, expected 1.0", sum);
}

#[test]
fn test_rope() {
    let client = get_client();
    let head_dim = 8;
    let position = 5u32;
    let freq_base = 10000.0f32;

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut expected = input.clone();
    cpu_rope(&mut expected, head_dim, position as usize, freq_base);

    let data_gpu = upload(&client, &input);

    unsafe {
        nnx_cubecl::rope::rope_kernel::launch::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&data_gpu, head_dim, 1),
            ScalarArg::new(0u32),
            ScalarArg::new(head_dim as u32),
            ScalarArg::new(position),
            ScalarArg::new(freq_base),
        );
    }

    let result = download(&client, &data_gpu);
    assert_close(&result, &expected, 1e-4, "rope");
}
