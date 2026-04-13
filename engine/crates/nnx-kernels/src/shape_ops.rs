//! Shape manipulation operations.
//!
//! These operate on flat f32 buffers with shape metadata passed as parameters.
//! The caller is responsible for computing correct output shapes.

/// Transpose a 2D matrix: [rows, cols] → [cols, rows]
pub fn transpose_2d_f32(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(input.len(), rows * cols);
    assert_eq!(output.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

/// Gather: output[i] = input[indices[i]] along the given axis.
/// For a 2D [rows, cols] with axis=0: gather rows by index.
pub fn gather_f32(
    input: &[f32],
    indices: &[usize],
    output: &mut [f32],
    axis_size: usize,    // size of the axis being gathered from
    inner_size: usize,   // product of dimensions after the axis
) {
    for (i, &idx) in indices.iter().enumerate() {
        let src_offset = idx * inner_size;
        let dst_offset = i * inner_size;
        output[dst_offset..dst_offset + inner_size]
            .copy_from_slice(&input[src_offset..src_offset + inner_size]);
    }
}

/// Concat two buffers along the "last" dimension.
/// For 2D: a is [rows, cols_a], b is [rows, cols_b], output is [rows, cols_a + cols_b]
pub fn concat_inner_f32(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    num_outer: usize,
    inner_a: usize,
    inner_b: usize,
) {
    let inner_out = inner_a + inner_b;
    for i in 0..num_outer {
        let a_start = i * inner_a;
        let b_start = i * inner_b;
        let o_start = i * inner_out;
        output[o_start..o_start + inner_a].copy_from_slice(&a[a_start..a_start + inner_a]);
        output[o_start + inner_a..o_start + inner_out].copy_from_slice(&b[b_start..b_start + inner_b]);
    }
}

/// Slice: extract a contiguous range along one dimension.
/// For 1D: output = input[start..end]
pub fn slice_1d_f32(input: &[f32], output: &mut [f32], start: usize, end: usize) {
    let len = end - start;
    assert_eq!(output.len(), len);
    output.copy_from_slice(&input[start..end]);
}

/// Pad with zeros: add pad_before zeros at start and pad_after at end.
pub fn pad_1d_f32(input: &[f32], output: &mut [f32], pad_before: usize, pad_after: usize) {
    let out_len = input.len() + pad_before + pad_after;
    assert_eq!(output.len(), out_len);
    output[..pad_before].fill(0.0);
    output[pad_before..pad_before + input.len()].copy_from_slice(input);
    output[pad_before + input.len()..].fill(0.0);
}

/// Repeat/tile: repeat input n times.
pub fn repeat_f32(input: &[f32], output: &mut [f32], repeats: usize) {
    assert_eq!(output.len(), input.len() * repeats);
    for r in 0..repeats {
        let offset = r * input.len();
        output[offset..offset + input.len()].copy_from_slice(input);
    }
}

/// Embedding lookup: output = weight[indices]
/// weight: [vocab_size, embed_dim], indices: [seq_len]
/// output: [seq_len, embed_dim]
pub fn embedding_f32(
    weight: &[f32],
    indices: &[u32],
    output: &mut [f32],
    embed_dim: usize,
) {
    for (i, &idx) in indices.iter().enumerate() {
        let src = idx as usize * embed_dim;
        let dst = i * embed_dim;
        output[dst..dst + embed_dim].copy_from_slice(&weight[src..src + embed_dim]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // 2x3
        let mut output = [0.0f32; 6]; // 3x2
        transpose_2d_f32(&input, &mut output, 2, 3);
        assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_gather() {
        let input = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0f32]; // 3 rows of 2
        let indices = [2, 0]; // pick rows 2 and 0
        let mut output = [0.0f32; 4]; // 2 rows of 2
        gather_f32(&input, &indices, &mut output, 3, 2);
        assert_eq!(output, [30.0, 31.0, 10.0, 11.0]);
    }

    #[test]
    fn test_embedding() {
        let weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6f32]; // vocab=3, dim=2
        let indices = [2u32, 0, 1];
        let mut output = [0.0f32; 6]; // 3 tokens × dim 2
        embedding_f32(&weight, &indices, &mut output, 2);
        assert_eq!(output, [0.5, 0.6, 0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_pad() {
        let input = [1.0, 2.0, 3.0f32];
        let mut output = [0.0f32; 7];
        pad_1d_f32(&input, &mut output, 2, 2);
        assert_eq!(output, [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
    }
}
