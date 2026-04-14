//! Matrix weight storage for transformer projections.
//!
//! Large linear weights can stay compact in memory and dispatch through the
//! quantized matvec kernel, while small vectors remain fully dequantized.

use nnx_kernels::matmul;
use nnx_quant::GGMLType;

/// Row-major matrix storage used by transformer projections.
#[derive(Debug, Clone)]
pub enum Matrix {
    Dense {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Quantized {
        data: Vec<u8>,
        dtype: GGMLType,
        rows: usize,
        cols: usize,
        row_bytes: usize,
    },
}

impl Matrix {
    /// Create dense row-major matrix storage.
    pub fn dense(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self::Dense { data, rows, cols }
    }

    /// Create compact matrix storage backed by raw GGUF tensor bytes.
    pub fn quantized(
        data: Vec<u8>,
        dtype: GGMLType,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        let row_bytes = row_bytes(dtype, cols)?;
        let expected_bytes = rows * row_bytes;
        if data.len() != expected_bytes {
            return Err(format!(
                "matrix bytes {} do not match expected {} for {}x{} {}",
                data.len(),
                expected_bytes,
                rows,
                cols,
                dtype
            ));
        }

        Ok(Self::Quantized {
            data,
            dtype,
            rows,
            cols,
            row_bytes,
        })
    }

    /// Multiply `self` by vector `x`, writing the output to `y`.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Self::Dense { data, rows, cols } => {
                debug_assert_eq!(x.len(), *cols);
                debug_assert_eq!(y.len(), *rows);
                matmul::matvec_f32(data, x, y, *rows, *cols);
            }
            Self::Quantized {
                data,
                dtype,
                rows,
                cols,
                row_bytes,
            } => {
                debug_assert_eq!(x.len(), *cols);
                debug_assert_eq!(y.len(), *rows);
                matmul::matvec_quantized(
                    |row| {
                        let start = row * row_bytes;
                        &data[start..start + row_bytes]
                    },
                    x,
                    y,
                    *rows,
                    *cols,
                    *dtype,
                );
            }
        }
    }

    /// Apply this matrix to a batch of input rows.
    ///
    /// `inputs` is `[batch_size, cols]` row-major and `outputs` is
    /// `[batch_size, rows]` row-major.
    pub fn matmul_input_rows(&self, inputs: &[f32], batch_size: usize, outputs: &mut [f32]) {
        match self {
            Self::Dense { data, rows, cols } => {
                debug_assert_eq!(inputs.len(), batch_size * *cols);
                debug_assert_eq!(outputs.len(), batch_size * *rows);

                // `matmul_f32` expects rhs as [K, N]. Dense weights are stored as
                // [N, K], so build a transposed scratch view for batched prefill.
                let mut rhs = vec![0.0f32; data.len()];
                for out_idx in 0..*rows {
                    for in_idx in 0..*cols {
                        rhs[in_idx * *rows + out_idx] = data[out_idx * *cols + in_idx];
                    }
                }
                matmul::matmul_f32(inputs, &rhs, outputs, batch_size, *cols, *rows);
            }
            Self::Quantized { rows, cols, .. } => {
                debug_assert_eq!(inputs.len(), batch_size * *cols);
                debug_assert_eq!(outputs.len(), batch_size * *rows);

                let mut row_buf = vec![0.0f32; *cols];
                for batch_idx in 0..batch_size {
                    let input = &inputs[batch_idx * *cols..(batch_idx + 1) * *cols];
                    let output = &mut outputs[batch_idx * *rows..(batch_idx + 1) * *rows];
                    if let Self::Quantized {
                        data,
                        dtype,
                        row_bytes,
                        ..
                    } = self
                    {
                        matmul::matvec_quantized_with_scratch(
                            |row| {
                                let start = row * row_bytes;
                                &data[start..start + row_bytes]
                            },
                            input,
                            output,
                            *rows,
                            *cols,
                            *dtype,
                            &mut row_buf,
                        );
                    }
                }
            }
        }
    }

    /// Copy a single logical row into `output`, dequantizing if needed.
    pub fn copy_row_to(&self, row: usize, output: &mut [f32]) {
        match self {
            Self::Dense { data, rows, cols } => {
                debug_assert!(row < *rows);
                debug_assert_eq!(output.len(), *cols);
                let start = row * *cols;
                output.copy_from_slice(&data[start..start + *cols]);
            }
            Self::Quantized {
                data,
                dtype,
                rows,
                cols,
                row_bytes,
            } => {
                debug_assert!(row < *rows);
                debug_assert_eq!(output.len(), *cols);
                let start = row * *row_bytes;
                nnx_quant::dequant::dequantize(&data[start..start + *row_bytes], *dtype, output);
            }
        }
    }

    /// Memory used by this matrix storage.
    pub fn storage_bytes(&self) -> usize {
        match self {
            Self::Dense { data, .. } => data.len() * std::mem::size_of::<f32>(),
            Self::Quantized { data, .. } => data.len(),
        }
    }
}

pub(crate) fn row_bytes(dtype: GGMLType, cols: usize) -> Result<usize, String> {
    let block_numel = dtype.block_numel();
    let block_bytes = dtype.block_size_bytes();
    if block_numel == 0 || block_bytes == 0 {
        return Err(format!("unsupported matrix dtype {}", dtype));
    }

    let num_blocks = (cols + block_numel - 1) / block_numel;
    Ok(num_blocks * block_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnx_quant::blocks::BlockQ8_0;

    #[test]
    fn test_dense_matvec() {
        let matrix = Matrix::dense(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let mut output = vec![0.0; 2];
        matrix.matvec(&[1.0, 1.0], &mut output);
        assert_eq!(output, vec![3.0, 7.0]);
    }

    #[test]
    fn test_quantized_q8_matvec() {
        let row0 = BlockQ8_0 {
            scale: half::f16::from_f32(1.0),
            quants: [
                1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        };
        let row1 = BlockQ8_0 {
            scale: half::f16::from_f32(1.0),
            quants: [
                5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        };

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&row0.scale.to_bits().to_le_bytes());
        bytes.extend(row0.quants.iter().map(|v| *v as u8));
        bytes.extend_from_slice(&row1.scale.to_bits().to_le_bytes());
        bytes.extend(row1.quants.iter().map(|v| *v as u8));

        let matrix = Matrix::quantized(bytes, GGMLType::Q8_0, 2, 32).unwrap();
        let mut output = vec![0.0; 2];
        let mut input = vec![0.0; 32];
        input[0] = 1.0;
        input[1] = 1.0;
        input[2] = 1.0;
        input[3] = 1.0;

        matrix.matvec(&input, &mut output);

        assert_eq!(output, vec![10.0, 26.0]);
    }

    #[test]
    fn test_quantized_q8_matmul_input_rows() {
        let row0 = BlockQ8_0 {
            scale: half::f16::from_f32(1.0),
            quants: [
                1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        };
        let row1 = BlockQ8_0 {
            scale: half::f16::from_f32(1.0),
            quants: [
                5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        };

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&row0.scale.to_bits().to_le_bytes());
        bytes.extend(row0.quants.iter().map(|v| *v as u8));
        bytes.extend_from_slice(&row1.scale.to_bits().to_le_bytes());
        bytes.extend(row1.quants.iter().map(|v| *v as u8));

        let matrix = Matrix::quantized(bytes, GGMLType::Q8_0, 2, 32).unwrap();
        let mut inputs = vec![0.0; 64];
        inputs[0] = 1.0;
        inputs[1] = 1.0;
        inputs[2] = 1.0;
        inputs[3] = 1.0;
        inputs[32] = 1.0;
        inputs[33] = 2.0;
        inputs[34] = 3.0;
        inputs[35] = 4.0;

        let mut outputs = vec![0.0; 4];
        matrix.matmul_input_rows(&inputs, 2, &mut outputs);

        assert_eq!(outputs, vec![10.0, 26.0, 30.0, 70.0]);
    }
}
