//! GPU activation quantization kernels for Q8_0 round-trip.
//!
//! These kernels quantize intermediate f32 activations (like attention Q, scores)
//! into Q8_0 blocks and dequantize them back, matching the CPU Q8_0 format:
//! - Block size: 32 elements
//! - Storage: f16 scale + 32 signed i8 quants
//! - Total: 34 bytes per block (2 + 32)

use cubecl::prelude::*;

/// Logical values per Q8_0 block.
const BLOCK_NUMEL: u32 = 32;

/// Quantize a flat f32 buffer into Q8_0 format (scale + quantized values).
///
/// For each block of 32 f32 values:
/// - Compute max absolute value
/// - Store scale as f16 (scale = max_abs / 127.0)
/// - Quantize each value to signed i8 via round(value / scale)
///
/// Layout: Scales stored separately for GPU efficiency.
/// - `scales`: One f32 per block (converted from f16 on write)
/// - `quants`: Packed u32 words, 8 words per block (32 bytes = 8 x 4-byte words)
#[cube(launch)]
pub fn quantize_f32_to_q8_0_kernel(
    input: &Array<f32>,
    scales: &mut Array<f32>,
    quants: &mut Array<u32>,
    numel: u32,
) {
    let block_id = ABSOLUTE_POS;
    let num_blocks = (numel + BLOCK_NUMEL - 1u32) / BLOCK_NUMEL;

    if block_id < num_blocks {
        let base = block_id * BLOCK_NUMEL;
        let block_end = Min::min(base + BLOCK_NUMEL, numel);

        // Find max absolute value in this block
        let mut max_abs = 0.0f32;
        for i in base..block_end {
            let val = f32::abs(input[i]);
            if val > max_abs {
                max_abs = val;
            }
        }

        // Compute scale (matching CPU Q8_0 encoding)
        let scale = if max_abs == 0.0f32 {
            0.0f32.into()
        } else {
            max_abs / 127.0f32
        };
        scales[block_id] = scale;

        // Quantize and pack 32 i8 values into 8 u32 words
        let inv_scale = if scale == 0.0f32 { 0.0f32.into() } else { 1.0f32 / scale };
        for word_idx in 0u32..8u32 {
            let mut packed = 0u32;
            for byte_idx in 0u32..4u32 {
                let elem_idx = base + word_idx * 4u32 + byte_idx;
                let quant = if elem_idx < block_end {
                    let val = input[elem_idx];
                    let q = f32::round(val * inv_scale);
                    let clamped = f32::clamp(q, -128.0f32, 127.0f32) as i32;
                    (clamped & 0xff) as u32
                } else {
                    0u32.into()
                };
                packed = packed | (quant << (byte_idx * 8u32));
            }
            quants[block_id * 8u32 + word_idx] = packed;
        }
    }
}

/// Dequantize Q8_0 blocks back to f32.
///
/// Reads the packed scale + quant representation and reconstructs f32 values
/// via: `output[i] = scale * (signed_quant[i])`.
#[cube(launch)]
pub fn dequantize_q8_0_to_f32_kernel(
    scales: &Array<f32>,
    quants: &Array<u32>,
    output: &mut Array<f32>,
    numel: u32,
) {
    let block_id = ABSOLUTE_POS;
    let num_blocks = (numel + BLOCK_NUMEL - 1u32) / BLOCK_NUMEL;

    if block_id < num_blocks {
        let base = block_id * BLOCK_NUMEL;
        let block_end = Min::min(base + BLOCK_NUMEL, numel);
        let scale = scales[block_id];

        // Unpack 8 u32 words into 32 signed i8 values
        for word_idx in 0u32..8u32 {
            let packed = quants[block_id * 8u32 + word_idx];
            for byte_idx in 0u32..4u32 {
                let elem_idx = base + word_idx * 4u32 + byte_idx;
                if elem_idx < block_end {
                    let byte_val = (packed >> (byte_idx * 8u32)) & 0xffu32;
                    // Sign-extend i8 to i32
                    let signed = if byte_val >= 128u32 {
                        (byte_val as i32) - 256i32
                    } else {
                        byte_val as i32
                    };
                    output[elem_idx] = scale * (signed as f32);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_size_constants() {
        assert_eq!(BLOCK_NUMEL, 32);
    }
}
