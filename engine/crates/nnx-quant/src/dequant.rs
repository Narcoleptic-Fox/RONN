//! Dequantization kernels — convert quantized blocks to f32.

use crate::blocks::*;
use crate::types::GGMLType;

/// Dequantize a Q4_0 block (32 values).
pub fn dequantize_q4_0(block: &BlockQ4_0, output: &mut [f32; 32]) {
    let scale = block.scale.to_f32();
    for i in 0..16 {
        let byte = block.quants[i];
        output[i * 2] = scale * ((byte & 0x0F) as i32 - 8) as f32;
        output[i * 2 + 1] = scale * (((byte >> 4) & 0x0F) as i32 - 8) as f32;
    }
}

/// Dequantize a Q8_0 block (32 values).
pub fn dequantize_q8_0(block: &BlockQ8_0, output: &mut [f32; 32]) {
    let scale = block.scale.to_f32();
    for i in 0..32 {
        output[i] = scale * block.quants[i] as f32;
    }
}

/// Dequantize a Q4_K block (256 values).
///
/// Q4_K uses super-block scale/min with 8 sub-blocks of 32 values.
/// Sub-block scales and mins are packed into 12 bytes.
pub fn dequantize_q4_k(block: &BlockQ4K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    // Unpack the 8 sub-block scales and mins from the packed 12-byte format
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..4 {
        sc[j] = block.scales[j] & 63;
        mn[j] = block.scales[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (block.scales[j + 4] & 0x0F) | ((block.scales[j - 4] >> 6) << 4);
        mn[j] = (block.scales[j + 4] >> 4) | ((block.scales[j] >> 6) << 4);
    }

    // Dequantize: 128 bytes hold 256 nibbles
    // First 128 values come from low nibbles, next 128 from high nibbles
    for j in 0..128 {
        let sub_lo = j / 32;         // sub-block for low-nibble half
        let sub_hi = sub_lo + 4;     // sub-block for high-nibble half
        let lo_nibble = (block.quants[j] & 0x0F) as f32;
        let hi_nibble = ((block.quants[j] >> 4) & 0x0F) as f32;
        output[j] = d * sc[sub_lo] as f32 * lo_nibble - dmin * mn[sub_lo] as f32;
        output[j + 128] = d * sc[sub_hi] as f32 * hi_nibble - dmin * mn[sub_hi] as f32;
    }
}

/// Dequantize a Q6_K block (256 values).
///
/// Q6_K: 6-bit quantization with per-sub-block i8 scales.
pub fn dequantize_q6_k(block: &BlockQ6K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();

    // 256 values: ql has low 4 bits, qh has high 2 bits
    // 16 sub-blocks of 16 values, each with an i8 scale
    for idx in 0..256 {
        let sub = idx / 16; // which sub-block (0..15)
        let ql_idx = idx / 2;
        let qh_idx = idx / 4;

        // Extract low 4 bits
        let lo = if idx % 2 == 0 {
            (block.ql[ql_idx] & 0x0F) as i32
        } else {
            ((block.ql[ql_idx] >> 4) & 0x0F) as i32
        };

        // Extract high 2 bits
        let shift = (idx % 4) * 2;
        let hi = ((block.qh[qh_idx] >> shift) & 0x03) as i32;

        // Combine to 6-bit value (0..63), then center to signed (-32..31)
        let q = (lo | (hi << 4)) - 32;

        output[idx] = d * block.scales[sub] as f32 * q as f32;
    }
}

/// Dequantize a Q5_K block (256 values).
///
/// Q5_K: 5-bit quantization. Low 4 bits in qs[], high bit in qh[].
/// Same scale packing as Q4_K.
pub fn dequantize_q5_k(block: &BlockQ5K, output: &mut [f32; 256]) {
    let d = block.d.to_f32();
    let dmin = block.dmin.to_f32();

    // Unpack sub-block scales and mins (same as Q4_K)
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..4 {
        sc[j] = block.scales[j] & 63;
        mn[j] = block.scales[j + 4] & 63;
    }
    for j in 4..8 {
        sc[j] = (block.scales[j + 4] & 0x0F) | ((block.scales[j - 4] >> 6) << 4);
        mn[j] = (block.scales[j + 4] >> 4) | ((block.scales[j] >> 6) << 4);
    }

    // Low nibbles (first 128 values) and high nibbles (next 128)
    for j in 0..128 {
        let sub_lo = j / 32;
        let sub_hi = sub_lo + 4;

        let lo_nibble = (block.qs[j] & 0x0F) as u32;
        let hi_nibble = ((block.qs[j] >> 4) & 0x0F) as u32;

        // Extract the 5th bit from qh[]
        let qh_byte = block.qh[j / 8];
        let bit_lo = ((qh_byte >> (j % 8)) & 1) as u32;
        let qh_byte2 = block.qh[(j + 128) / 8];
        let bit_hi = ((qh_byte2 >> ((j + 128) % 8)) & 1) as u32;

        let val_lo = lo_nibble | (bit_lo << 4); // 5-bit value
        let val_hi = hi_nibble | (bit_hi << 4);

        output[j] = d * sc[sub_lo] as f32 * val_lo as f32 - dmin * mn[sub_lo] as f32;
        output[j + 128] = d * sc[sub_hi] as f32 * val_hi as f32 - dmin * mn[sub_hi] as f32;
    }
}

/// Dequantize BF16 values to f32.
fn dequantize_bf16(data: &[u8], output: &mut [f32]) -> usize {
    let count = (data.len() / 2).min(output.len());
    for i in 0..count {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        // BF16 → f32: just shift left by 16 bits
        output[i] = f32::from_bits((bits as u32) << 16);
    }
    count
}

/// Dequantize a raw byte buffer of the given type into f32.
/// Returns the number of f32 values written.
pub fn dequantize(data: &[u8], dtype: GGMLType, output: &mut [f32]) -> usize {
    match dtype {
        GGMLType::F32 => {
            let count = (data.len() / 4).min(output.len());
            for i in 0..count {
                output[i] = f32::from_le_bytes([
                    data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3],
                ]);
            }
            count
        }
        GGMLType::F16 => {
            let count = (data.len() / 2).min(output.len());
            for i in 0..count {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                output[i] = half::f16::from_bits(bits).to_f32();
            }
            count
        }
        GGMLType::BF16 => dequantize_bf16(data, output),
        GGMLType::Q4_0 => dequant_blocks::<BlockQ4_0, 32>(data, output, dequantize_q4_0),
        GGMLType::Q8_0 => dequant_blocks::<BlockQ8_0, 32>(data, output, dequantize_q8_0),
        GGMLType::Q4K => dequant_blocks::<BlockQ4K, 256>(data, output, dequantize_q4_k),
        GGMLType::Q5K => dequant_blocks::<BlockQ5K, 256>(data, output, dequantize_q5_k),
        GGMLType::Q6K => dequant_blocks::<BlockQ6K, 256>(data, output, dequantize_q6_k),
        _ => 0,
    }
}

/// Generic block dequantization helper.
fn dequant_blocks<B: Copy, const N: usize>(
    data: &[u8],
    output: &mut [f32],
    dequant_fn: fn(&B, &mut [f32; N]),
) -> usize {
    let block_bytes = std::mem::size_of::<B>();
    let num_blocks = data.len() / block_bytes;
    let mut written = 0;
    for b in 0..num_blocks {
        if written + N > output.len() { break; }
        let offset = b * block_bytes;
        let block = unsafe { &*(data[offset..].as_ptr() as *const B) };
        let out: &mut [f32; N] = (&mut output[written..written + N]).try_into().unwrap();
        dequant_fn(block, out);
        written += N;
    }
    written
}

/// Dequantize an entire tensor's worth of data, allocating the output.
pub fn dequantize_alloc(data: &[u8], dtype: GGMLType, numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    dequantize(data, dtype, &mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_q4_0_zero_scale() {
        let block = BlockQ4_0 {
            scale: half::f16::from_f32(0.0),
            quants: [0x88; 16],
        };
        let mut output = [0.0f32; 32];
        dequantize_q4_0(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q8_0_identity() {
        let mut quants = [0i8; 32];
        for i in 0..32 { quants[i] = i as i8; }
        let block = BlockQ8_0 { scale: half::f16::from_f32(1.0), quants };
        let mut output = [0.0f32; 32];
        dequantize_q8_0(&block, &mut output);
        for i in 0..32 { assert!((output[i] - i as f32).abs() < 0.01); }
    }

    #[test]
    fn test_dequant_q4_k_zero() {
        let block = BlockQ4K {
            d: half::f16::from_f32(0.0),
            dmin: half::f16::from_f32(0.0),
            scales: [0; 12],
            quants: [0; 128],
        };
        let mut output = [0.0f32; 256];
        dequantize_q4_k(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q6_k_zero() {
        let block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: half::f16::from_f32(0.0),
        };
        let mut output = [0.0f32; 256];
        dequantize_q6_k(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_bf16() {
        // BF16 for 1.0 = 0x3F80
        let data = [0x80u8, 0x3F, 0x00, 0x40]; // 1.0 and 2.0 in BF16
        let mut output = [0.0f32; 2];
        let n = dequantize_bf16(&data, &mut output);
        assert_eq!(n, 2);
        assert!((output[0] - 1.0).abs() < 1e-2);
        assert!((output[1] - 2.0).abs() < 1e-2);
    }

    #[test]
    fn test_dequant_f32_passthrough() {
        let input: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut output = [0.0f32; 3];
        let n = dequantize(&input, GGMLType::F32, &mut output);
        assert_eq!(n, 3);
        assert_eq!(output, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dequant_alloc() {
        let input: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let output = dequantize_alloc(&input, GGMLType::F32, 2);
        assert_eq!(output, vec![1.0, 2.0]);
    }
}
