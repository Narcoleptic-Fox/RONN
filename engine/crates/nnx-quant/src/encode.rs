//! Quantization encoders for GGML-compatible matrix payloads.

use crate::GGMLType;
use crate::dequant::dequantize;

const BLOCK_NUMEL: usize = 32;

/// Compute the packed row width in bytes for a quantized matrix.
pub fn row_bytes(dtype: GGMLType, cols: usize) -> Result<usize, String> {
    let block_numel = dtype.block_numel();
    let block_bytes = dtype.block_size_bytes();
    if block_numel == 0 || block_bytes == 0 {
        return Err(format!("unsupported matrix dtype {}", dtype));
    }

    Ok(cols.div_ceil(block_numel) * block_bytes)
}

/// Quantize a row-major `f32` matrix into GGML block bytes.
pub fn quantize_matrix(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    dtype: GGMLType,
) -> Result<Vec<u8>, String> {
    if matrix.len() != rows * cols {
        return Err(format!(
            "matrix has {} elements, expected {} for {}x{}",
            matrix.len(),
            rows * cols,
            rows,
            cols
        ));
    }

    match dtype {
        GGMLType::Q8_0 => quantize_q8_0_matrix(matrix, rows, cols),
        GGMLType::Q4_0 => quantize_q4_0_matrix(matrix, rows, cols),
        other => Err(format!("quantization encoder for {} is not implemented", other)),
    }
}

/// Quantize a row-major `f32` matrix and immediately dequantize it back in place.
///
/// This is useful for opt-in activation quantization paths that want to pass
/// intermediate buffers through a compact representation before later kernels
/// learn to consume quantized activations directly.
pub fn roundtrip_matrix_in_place(
    matrix: &mut [f32],
    rows: usize,
    cols: usize,
    dtype: GGMLType,
) -> Result<(), String> {
    if matrix.len() != rows * cols {
        return Err(format!(
            "matrix has {} elements, expected {} for {}x{}",
            matrix.len(),
            rows * cols,
            rows,
            cols
        ));
    }

    let quantized = quantize_matrix(matrix, rows, cols, dtype)?;
    let row_stride = row_bytes(dtype, cols)?;
    for row in 0..rows {
        let start = row * row_stride;
        let end = start + row_stride;
        let dst = &mut matrix[row * cols..(row + 1) * cols];
        dequantize(&quantized[start..end], dtype, dst);
    }

    Ok(())
}

fn quantize_q8_0_matrix(matrix: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    let bytes_per_row = row_bytes(GGMLType::Q8_0, cols)?;
    let mut bytes = Vec::with_capacity(rows * bytes_per_row);

    for row in 0..rows {
        let row_data = &matrix[row * cols..(row + 1) * cols];
        for block_start in (0..cols).step_by(BLOCK_NUMEL) {
            let block_end = (block_start + BLOCK_NUMEL).min(cols);
            let block = &row_data[block_start..block_end];
            bytes.extend_from_slice(&quantize_q8_0_block(block));
        }
    }

    Ok(bytes)
}

fn quantize_q4_0_matrix(matrix: &[f32], rows: usize, cols: usize) -> Result<Vec<u8>, String> {
    let bytes_per_row = row_bytes(GGMLType::Q4_0, cols)?;
    let mut bytes = Vec::with_capacity(rows * bytes_per_row);

    for row in 0..rows {
        let row_data = &matrix[row * cols..(row + 1) * cols];
        for block_start in (0..cols).step_by(BLOCK_NUMEL) {
            let block_end = (block_start + BLOCK_NUMEL).min(cols);
            let block = &row_data[block_start..block_end];
            bytes.extend_from_slice(&quantize_q4_0_block(block));
        }
    }

    Ok(bytes)
}

fn quantize_q8_0_block(values: &[f32]) -> Vec<u8> {
    let mut quants = [0i8; BLOCK_NUMEL];
    let max_abs = values.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
    let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };

    for (index, quant) in quants.iter_mut().enumerate() {
        let value = values.get(index).copied().unwrap_or(0.0);
        *quant = if scale == 0.0 {
            0
        } else {
            (value / scale).round().clamp(-128.0, 127.0) as i8
        };
    }

    let mut bytes = Vec::with_capacity(34);
    bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
    bytes.extend(quants.iter().map(|quant| *quant as u8));
    bytes
}

fn quantize_q4_0_block(values: &[f32]) -> Vec<u8> {
    let mut quants = [0i8; BLOCK_NUMEL];
    let max_abs = values.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
    let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 7.0 };

    for (index, quant) in quants.iter_mut().enumerate() {
        let value = values.get(index).copied().unwrap_or(0.0);
        *quant = if scale == 0.0 {
            0
        } else {
            (value / scale).round().clamp(-8.0, 7.0) as i8
        };
    }

    let mut bytes = Vec::with_capacity(18);
    bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
    for pair in 0..(BLOCK_NUMEL / 2) {
        let low = (quants[pair * 2] + 8) as u8;
        let high = (quants[pair * 2 + 1] + 8) as u8;
        bytes.push(low | (high << 4));
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix(rows: usize, cols: usize) -> Vec<f32> {
        let mut matrix = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                let centered = ((row * 13 + col * 7 + 3) % 29) as f32 - 14.0;
                let swing = ((row * 5 + col * 11 + 1) % 9) as f32 - 4.0;
                matrix.push(centered * 0.17 + swing * 0.09);
            }
        }
        matrix
    }

    fn dequantize_rows(bytes: &[u8], rows: usize, cols: usize, dtype: GGMLType) -> Vec<f32> {
        let row_stride = row_bytes(dtype, cols).unwrap();
        let mut output = vec![0.0f32; rows * cols];
        for row in 0..rows {
            let start = row * row_stride;
            let end = start + row_stride;
            let dst = &mut output[row * cols..(row + 1) * cols];
            dequantize(&bytes[start..end], dtype, dst);
        }
        output
    }

    fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_quantize_q8_0_roundtrip_stays_close() {
        let rows = 3;
        let cols = 37;
        let matrix = sample_matrix(rows, cols);

        let quantized = quantize_matrix(&matrix, rows, cols, GGMLType::Q8_0).unwrap();
        let restored = dequantize_rows(&quantized, rows, cols, GGMLType::Q8_0);

        assert_eq!(quantized.len(), rows * row_bytes(GGMLType::Q8_0, cols).unwrap());
        assert!(max_abs_error(&matrix, &restored) < 0.05);
    }

    #[test]
    fn test_quantize_q4_0_roundtrip_stays_reasonable() {
        let rows = 4;
        let cols = 35;
        let matrix = sample_matrix(rows, cols);

        let quantized = quantize_matrix(&matrix, rows, cols, GGMLType::Q4_0).unwrap();
        let restored = dequantize_rows(&quantized, rows, cols, GGMLType::Q4_0);

        assert_eq!(quantized.len(), rows * row_bytes(GGMLType::Q4_0, cols).unwrap());
        assert!(max_abs_error(&matrix, &restored) < 0.5);
    }

    #[test]
    fn test_quantize_matrix_rejects_shape_mismatch() {
        let error = quantize_matrix(&[1.0, 2.0, 3.0], 2, 2, GGMLType::Q8_0)
            .expect_err("shape mismatch should error");
        assert!(error.contains("expected 4"));
    }

    #[test]
    fn test_roundtrip_matrix_in_place_q8_0_stays_close() {
        let rows = 2;
        let cols = 33;
        let original = sample_matrix(rows, cols);
        let mut roundtripped = original.clone();

        roundtrip_matrix_in_place(&mut roundtripped, rows, cols, GGMLType::Q8_0).unwrap();

        assert!(max_abs_error(&original, &roundtripped) < 0.05);
    }
}