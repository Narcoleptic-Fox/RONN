//! Quantized matrix-vector multiply kernels.
//!
//! CubeCL kernels operate on typed arrays, so raw GGML row bytes are repacked
//! on the host into per-block `f32` scales plus packed `u32` payload words.
//! The GPU then dequantizes each block on the fly while accumulating the dot
//! product for one output row per cube.

use cubecl::prelude::*;

/// Logical values per GGML Q4_0/Q8_0 block.
pub const QUANT_BLOCK_NUMEL: usize = 32;
const QUANT_BLOCK_NUMEL_U32: u32 = 32;
/// Bytes per GGML Q8_0 block: 2-byte f16 scale + 32 signed quants.
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Bytes per GGML Q4_0 block: 2-byte f16 scale + 16 packed nibble bytes.
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Packed `u32` words per Q8_0 block after removing the scale bytes.
pub const Q8_0_DATA_WORDS: usize = 8;
const Q8_0_DATA_WORDS_U32: u32 = 8;
/// Packed `u32` words per Q4_0 block after removing the scale bytes.
pub const Q4_0_DATA_WORDS: usize = 4;
const Q4_0_DATA_WORDS_U32: u32 = 4;

fn encode_f32_slice(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_u32_slice(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32_slice(bytes: &[u8]) -> Vec<f32> {
    f32::from_bytes(bytes).to_vec()
}

fn pack_block_bytes_to_words(
    data: &[u8],
    bytes_per_block: usize,
) -> Result<(Vec<f32>, Vec<u32>), String> {
    if data.len() % bytes_per_block != 0 {
        return Err(format!(
            "quantized byte length {} is not a multiple of block size {}",
            data.len(),
            bytes_per_block
        ));
    }

    let blocks = data.len() / bytes_per_block;
    let payload_bytes = bytes_per_block - std::mem::size_of::<u16>();
    let words_per_block = payload_bytes / std::mem::size_of::<u32>();

    let mut scales = Vec::with_capacity(blocks);
    let mut words = Vec::with_capacity(blocks * words_per_block);

    for block in data.chunks_exact(bytes_per_block) {
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        scales.push(half::f16::from_bits(scale_bits).to_f32());

        for payload in block[2..].chunks_exact(std::mem::size_of::<u32>()) {
            words.push(u32::from_le_bytes([
                payload[0], payload[1], payload[2], payload[3],
            ]));
        }
    }

    Ok((scales, words))
}

/// Repack GGML Q8_0 row bytes into host-uploadable scale and payload buffers.
pub fn pack_q8_0_weights(data: &[u8]) -> Result<(Vec<f32>, Vec<u32>), String> {
    pack_block_bytes_to_words(data, Q8_0_BLOCK_BYTES)
}

/// Repack GGML Q4_0 row bytes into host-uploadable scale and payload buffers.
pub fn pack_q4_0_weights(data: &[u8]) -> Result<(Vec<f32>, Vec<u32>), String> {
    pack_block_bytes_to_words(data, Q4_0_BLOCK_BYTES)
}

/// Launch a Q8_0 matrix-vector multiply from raw GGML bytes.
pub fn q8_0_matvec<R: Runtime>(
    client: &ComputeClient<<R as Runtime>::Server>,
    weights: &[u8],
    vector: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let blocks_per_row = cols.div_ceil(QUANT_BLOCK_NUMEL);
    let expected_bytes = rows * blocks_per_row * Q8_0_BLOCK_BYTES;
    if weights.len() != expected_bytes {
        return Err(format!(
            "Q8_0 weights len {} does not match expected {} for {}x{}",
            weights.len(),
            expected_bytes,
            rows,
            cols
        ));
    }

    let (scales, quants) = pack_q8_0_weights(weights)?;
    let output = vec![0.0f32; rows];

    let scale_bytes = encode_f32_slice(&scales);
    let quant_bytes = encode_u32_slice(&quants);
    let vector_bytes = encode_f32_slice(vector);
    let output_bytes = encode_f32_slice(&output);

    let scale_handle = client.create(scale_bytes.as_slice());
    let quant_handle = client.create(quant_bytes.as_slice());
    let vector_handle = client.create(vector_bytes.as_slice());
    let output_handle = client.create(output_bytes.as_slice());

    unsafe {
        q8_0_matvec_kernel::launch::<R>(
            client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&scale_handle, scales.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&quant_handle, quants.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&vector_handle, vector.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, output.len(), 1),
            ScalarArg::new(blocks_per_row as u32),
            ScalarArg::new(cols as u32),
        );
    }

    Ok(decode_f32_slice(&client.read_one(output_handle)))
}

/// Launch a Q4_0 matrix-vector multiply from raw GGML bytes.
pub fn q4_0_matvec<R: Runtime>(
    client: &ComputeClient<<R as Runtime>::Server>,
    weights: &[u8],
    vector: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let blocks_per_row = cols.div_ceil(QUANT_BLOCK_NUMEL);
    let expected_bytes = rows * blocks_per_row * Q4_0_BLOCK_BYTES;
    if weights.len() != expected_bytes {
        return Err(format!(
            "Q4_0 weights len {} does not match expected {} for {}x{}",
            weights.len(),
            expected_bytes,
            rows,
            cols
        ));
    }

    let (scales, quants) = pack_q4_0_weights(weights)?;
    let output = vec![0.0f32; rows];

    let scale_bytes = encode_f32_slice(&scales);
    let quant_bytes = encode_u32_slice(&quants);
    let vector_bytes = encode_f32_slice(vector);
    let output_bytes = encode_f32_slice(&output);

    let scale_handle = client.create(scale_bytes.as_slice());
    let quant_handle = client.create(quant_bytes.as_slice());
    let vector_handle = client.create(vector_bytes.as_slice());
    let output_handle = client.create(output_bytes.as_slice());

    unsafe {
        q4_0_matvec_kernel::launch::<R>(
            client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new(1, 1, 1),
            ArrayArg::from_raw_parts::<f32>(&scale_handle, scales.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&quant_handle, quants.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&vector_handle, vector.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, output.len(), 1),
            ScalarArg::new(blocks_per_row as u32),
            ScalarArg::new(cols as u32),
        );
    }

    Ok(decode_f32_slice(&client.read_one(output_handle)))
}

/// Q8_0 matrix-vector multiply: one cube computes one output row.
#[cube(launch)]
pub fn q8_0_matvec_kernel(
    scales: &Array<f32>,
    quants: &Array<u32>,
    vector: &Array<f32>,
    output: &mut Array<f32>,
    blocks_per_row: u32,
    cols: u32,
) {
    let row = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let mut sum = 0.0f32;
        let row_block_base = row * blocks_per_row;
        let row_quant_base = row_block_base * Q8_0_DATA_WORDS_U32;

        for block in 0..blocks_per_row {
            let scale = scales[row_block_base + block];
            let block_col_base = block * QUANT_BLOCK_NUMEL_U32;
            let block_quant_base = row_quant_base + block * Q8_0_DATA_WORDS_U32;

            for word_idx in 0..Q8_0_DATA_WORDS_U32 {
                let packed = quants[block_quant_base + word_idx];
                let word_col_base = block_col_base + word_idx * 4u32;

                for byte_idx in 0..4u32 {
                    let col = word_col_base + byte_idx;
                    if col < cols {
                        let quant_bits = (packed >> (byte_idx * 8u32)) & 0xFFu32;
                        let quant = if quant_bits >= 128u32 {
                            quant_bits as f32 - 256.0f32
                        } else {
                            quant_bits as f32
                        };
                        sum += (quant * scale) * vector[col];
                    }
                }
            }
        }

        output[row] = sum;
    }
}

/// Q4_0 matrix-vector multiply: one cube computes one output row.
#[cube(launch)]
pub fn q4_0_matvec_kernel(
    scales: &Array<f32>,
    quants: &Array<u32>,
    vector: &Array<f32>,
    output: &mut Array<f32>,
    blocks_per_row: u32,
    cols: u32,
) {
    let row = CUBE_POS_X;
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let mut sum = 0.0f32;
        let row_block_base = row * blocks_per_row;
        let row_quant_base = row_block_base * Q4_0_DATA_WORDS_U32;

        for block in 0..blocks_per_row {
            let scale = scales[row_block_base + block];
            let block_col_base = block * QUANT_BLOCK_NUMEL_U32;
            let block_quant_base = row_quant_base + block * Q4_0_DATA_WORDS_U32;

            for word_idx in 0..Q4_0_DATA_WORDS_U32 {
                let packed = quants[block_quant_base + word_idx];
                let word_col_base = block_col_base + word_idx * 8u32;

                for byte_idx in 0..4u32 {
                    let packed_byte = (packed >> (byte_idx * 8u32)) & 0xFFu32;
                    let low_col = word_col_base + byte_idx * 2u32;
                    if low_col < cols {
                        let low_quant = (packed_byte & 0x0Fu32) as f32 - 8.0f32;
                        sum += (low_quant * scale) * vector[low_col];
                    }

                    let high_col = low_col + 1u32;
                    if high_col < cols {
                        let high_quant = ((packed_byte >> 4u32) & 0x0Fu32) as f32 - 8.0f32;
                        sum += (high_quant * scale) * vector[high_col];
                    }
                }
            }
        }

        output[row] = sum;
    }
}
