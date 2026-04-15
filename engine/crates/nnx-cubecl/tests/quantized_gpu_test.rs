//! GPU correctness tests for quantized matvec kernels.
//!
//! Run with: `cargo test -p nnx-cubecl --features wgpu --test quantized_gpu_test`

#![cfg(feature = "wgpu")]

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use nnx_cubecl::quantized::{QUANT_BLOCK_NUMEL, q4_0_matvec, q8_0_matvec};
use nnx_quant::GGMLType;

type R = WgpuRuntime;
type Server = <R as Runtime>::Server;

fn get_client() -> ComputeClient<Server> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );

    for (idx, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() <= tol,
            "{label}: mismatch at [{idx}] gpu={av} cpu={bv} diff={}",
            (av - bv).abs(),
        );
    }
}

fn cpu_quantized_matvec(
    weights: &[u8],
    vector: &[f32],
    rows: usize,
    cols: usize,
    dtype: GGMLType,
) -> Vec<f32> {
    let block_numel = dtype.block_numel();
    let row_bytes = cols.div_ceil(block_numel) * dtype.block_size_bytes();
    let mut row = vec![0.0f32; cols];
    let mut out = vec![0.0f32; rows];

    for output_row in 0..rows {
        row.fill(0.0);
        let start = output_row * row_bytes;
        let end = start + row_bytes;
        nnx_quant::dequant::dequantize(&weights[start..end], dtype, &mut row);
        out[output_row] = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
    }

    out
}

fn build_matrix(rows: usize, cols: usize) -> Vec<f32> {
    let mut matrix = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let centered = ((row * 31 + col * 17 + row * col) % 41) as f32 - 20.0;
            let wave = ((row + col * 3) % 7) as f32 - 3.0;
            matrix[row * cols + col] = centered * 0.18 + wave * 0.07;
        }
    }
    matrix
}

fn build_vector(cols: usize) -> Vec<f32> {
    let mut vector = vec![0.0f32; cols];
    for col in 0..cols {
        let centered = ((col * 19 + 5) % 23) as f32 - 11.0;
        let swing = ((col * 7 + 3) % 5) as f32 - 2.0;
        vector[col] = centered * 0.11 + swing * 0.09;
    }
    vector
}

fn q8_0_row_bytes(scale: f32, quants: &[i8; QUANT_BLOCK_NUMEL]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(34);
    bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
    bytes.extend(quants.iter().map(|q| *q as u8));
    bytes
}

fn q4_0_row_bytes(scale: f32, quants: &[i8; QUANT_BLOCK_NUMEL]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(18);
    bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());

    for pair in 0..(QUANT_BLOCK_NUMEL / 2) {
        let low = quants[pair * 2];
        let high = quants[pair * 2 + 1];
        assert!(
            (-8..=7).contains(&low) && (-8..=7).contains(&high),
            "Q4_0 quants must be in [-8, 7]"
        );

        let low_nibble = (low + 8) as u8;
        let high_nibble = (high + 8) as u8;
        bytes.push(low_nibble | (high_nibble << 4));
    }

    bytes
}

fn quantize_q8_0_matrix(matrix: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let blocks_per_row = cols.div_ceil(QUANT_BLOCK_NUMEL);
    let mut bytes = Vec::with_capacity(rows * blocks_per_row * 34);

    for row in 0..rows {
        for block in 0..blocks_per_row {
            let mut quants = [0i8; QUANT_BLOCK_NUMEL];
            let mut max_abs = 0.0f32;

            for offset in 0..QUANT_BLOCK_NUMEL {
                let col = block * QUANT_BLOCK_NUMEL + offset;
                if col < cols {
                    max_abs = max_abs.max(matrix[row * cols + col].abs());
                }
            }

            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };

            for (offset, quant) in quants.iter_mut().enumerate() {
                let col = block * QUANT_BLOCK_NUMEL + offset;
                if col >= cols || scale == 0.0 {
                    *quant = 0;
                    continue;
                }

                let value = matrix[row * cols + col] / scale;
                *quant = value.round().clamp(-128.0, 127.0) as i8;
            }

            bytes.extend_from_slice(&q8_0_row_bytes(scale, &quants));
        }
    }

    bytes
}

fn quantize_q4_0_matrix(matrix: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let blocks_per_row = cols.div_ceil(QUANT_BLOCK_NUMEL);
    let mut bytes = Vec::with_capacity(rows * blocks_per_row * 18);

    for row in 0..rows {
        for block in 0..blocks_per_row {
            let mut quants = [0i8; QUANT_BLOCK_NUMEL];
            let mut max_abs = 0.0f32;

            for offset in 0..QUANT_BLOCK_NUMEL {
                let col = block * QUANT_BLOCK_NUMEL + offset;
                if col < cols {
                    max_abs = max_abs.max(matrix[row * cols + col].abs());
                }
            }

            let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 7.0 };

            for (offset, quant) in quants.iter_mut().enumerate() {
                let col = block * QUANT_BLOCK_NUMEL + offset;
                if col >= cols {
                    *quant = 0;
                    continue;
                }
                if scale == 0.0 {
                    *quant = 0;
                    continue;
                }

                let value = matrix[row * cols + col] / scale;
                *quant = value.round().clamp(-8.0, 7.0) as i8;
            }

            bytes.extend_from_slice(&q4_0_row_bytes(scale, &quants));
        }
    }

    bytes
}

fn run_q8_0_case(client: &ComputeClient<Server>, rows: usize, cols: usize, tol: f32) {
    let matrix = build_matrix(rows, cols);
    let vector = build_vector(cols);
    let weights = quantize_q8_0_matrix(&matrix, rows, cols);

    let gpu = q8_0_matvec::<R>(client, &weights, &vector, rows, cols).expect("Q8_0 GPU matvec");
    let cpu = cpu_quantized_matvec(&weights, &vector, rows, cols, GGMLType::Q8_0);

    assert_close(&gpu, &cpu, tol, &format!("Q8_0 {rows}x{cols}"));
}

fn run_q4_0_case(client: &ComputeClient<Server>, rows: usize, cols: usize, tol: f32) {
    let matrix = build_matrix(rows, cols);
    let vector = build_vector(cols);
    let weights = quantize_q4_0_matrix(&matrix, rows, cols);

    let gpu = q4_0_matvec::<R>(client, &weights, &vector, rows, cols).expect("Q4_0 GPU matvec");
    let cpu = cpu_quantized_matvec(&weights, &vector, rows, cols, GGMLType::Q4_0);

    assert_close(&gpu, &cpu, tol, &format!("Q4_0 {rows}x{cols}"));
}

#[test]
fn test_q8_0_single_block_matches_hand_computed_reference() {
    let client = get_client();

    let mut row0 = [0i8; QUANT_BLOCK_NUMEL];
    row0[0] = 2;
    row0[1] = -4;
    row0[2] = 6;
    row0[3] = -8;

    let mut row1 = [0i8; QUANT_BLOCK_NUMEL];
    row1[0] = -3;
    row1[1] = 5;
    row1[2] = -7;
    row1[3] = 9;

    let scale0 = 0.5f32;
    let scale1 = 0.25f32;
    let vector = [1.0f32, 2.0, -1.0, 0.5];

    let mut weights = q8_0_row_bytes(scale0, &row0);
    weights.extend_from_slice(&q8_0_row_bytes(scale1, &row1));

    let gpu = q8_0_matvec::<R>(&client, &weights, &vector, 2, 4).expect("Q8_0 hand case");
    let expected = vec![
        scale0 * (2.0 * 1.0 + -4.0 * 2.0 + 6.0 * -1.0 + -8.0 * 0.5),
        scale1 * (-3.0 * 1.0 + 5.0 * 2.0 + -7.0 * -1.0 + 9.0 * 0.5),
    ];

    assert_close(&gpu, &expected, 1e-5, "Q8_0 hand-computed");
}

#[test]
fn test_q4_0_single_block_matches_hand_computed_reference() {
    let client = get_client();

    let mut row0 = [0i8; QUANT_BLOCK_NUMEL];
    row0[0] = -8;
    row0[1] = -7;
    row0[2] = 6;
    row0[3] = 5;

    let mut row1 = [0i8; QUANT_BLOCK_NUMEL];
    row1[0] = 4;
    row1[1] = -3;
    row1[2] = 2;
    row1[3] = -1;

    let scale0 = 0.25f32;
    let scale1 = 0.5f32;
    let vector = [1.0f32, -2.0, 0.5, 3.0];

    let mut weights = q4_0_row_bytes(scale0, &row0);
    weights.extend_from_slice(&q4_0_row_bytes(scale1, &row1));

    let gpu = q4_0_matvec::<R>(&client, &weights, &vector, 2, 4).expect("Q4_0 hand case");
    let expected = vec![
        scale0 * (-8.0 * 1.0 + -7.0 * -2.0 + 6.0 * 0.5 + 5.0 * 3.0),
        scale1 * (4.0 * 1.0 + -3.0 * -2.0 + 2.0 * 0.5 + -1.0 * 3.0),
    ];

    assert_close(&gpu, &expected, 1e-5, "Q4_0 hand-computed");
}

#[test]
fn test_q8_0_matvec_across_matrix_sizes() {
    let client = get_client();

    for (rows, cols, tol) in [(3usize, 17usize, 1e-4f32), (19, 64, 2e-4), (48, 127, 5e-4)] {
        run_q8_0_case(&client, rows, cols, tol);
    }
}

#[test]
fn test_q4_0_matvec_across_matrix_sizes() {
    let client = get_client();

    for (rows, cols, tol) in [(3usize, 17usize, 1e-4f32), (19, 64, 2e-4), (48, 127, 5e-4)] {
        run_q4_0_case(&client, rows, cols, tol);
    }
}
