//! Masking operations: Triu, Tril, MaskedFill.
//!
//! Used for causal attention masks, padding masks, and conditional fills.

use nnx_core::error::EngineError;

/// Upper triangular mask: output[r,c] = (c >= r + diagonal)
///
/// For causal attention with `diagonal=1`:
///   [[true, false, false],
///    [true, true,  false],
///    [true, true,  true ]]
/// (tril with diagonal=0 is the complement)
pub fn triu_mask(output: &mut [bool], rows: usize, cols: usize, diagonal: isize) {
    assert_eq!(output.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            output[r * cols + c] = c as isize >= r as isize + diagonal;
        }
    }
}

/// Checked version of `triu_mask`.
pub fn triu_mask_checked(
    output: &mut [bool],
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> nnx_core::error::Result<()> {
    if output.len() != rows * cols {
        return Err(EngineError::ShapeMismatch(format!(
            "triu_mask: output.len()={} but rows*cols={}",
            output.len(),
            rows * cols
        )));
    }
    triu_mask(output, rows, cols, diagonal);
    Ok(())
}

/// Lower triangular mask: output[r,c] = (c <= r + diagonal)
pub fn tril_mask(output: &mut [bool], rows: usize, cols: usize, diagonal: isize) {
    assert_eq!(output.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            output[r * cols + c] = c as isize <= r as isize + diagonal;
        }
    }
}

/// Checked version of `tril_mask`.
pub fn tril_mask_checked(
    output: &mut [bool],
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> nnx_core::error::Result<()> {
    if output.len() != rows * cols {
        return Err(EngineError::ShapeMismatch(format!(
            "tril_mask: output.len()={} but rows*cols={}",
            output.len(),
            rows * cols
        )));
    }
    tril_mask(output, rows, cols, diagonal);
    Ok(())
}

/// Upper triangular fill: set elements above the diagonal to `value`.
/// Matrix is [rows, cols] in row-major order.
pub fn triu_f32(data: &mut [f32], rows: usize, cols: usize, diagonal: isize, value: f32) {
    assert_eq!(data.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if c as isize >= r as isize + diagonal {
                data[r * cols + c] = value;
            }
        }
    }
}

/// Checked version of `triu_f32`.
pub fn triu_f32_checked(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    diagonal: isize,
    value: f32,
) -> nnx_core::error::Result<()> {
    if data.len() != rows * cols {
        return Err(EngineError::ShapeMismatch(format!(
            "triu_f32: data.len()={} but rows*cols={}",
            data.len(),
            rows * cols
        )));
    }
    triu_f32(data, rows, cols, diagonal, value);
    Ok(())
}

/// Lower triangular fill: set elements below the diagonal to `value`.
pub fn tril_f32(data: &mut [f32], rows: usize, cols: usize, diagonal: isize, value: f32) {
    assert_eq!(data.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if c as isize <= r as isize + diagonal {
                data[r * cols + c] = value;
            }
        }
    }
}

/// Checked version of `tril_f32`.
pub fn tril_f32_checked(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    diagonal: isize,
    value: f32,
) -> nnx_core::error::Result<()> {
    if data.len() != rows * cols {
        return Err(EngineError::ShapeMismatch(format!(
            "tril_f32: data.len()={} but rows*cols={}",
            data.len(),
            rows * cols
        )));
    }
    tril_f32(data, rows, cols, diagonal, value);
    Ok(())
}

/// Masked fill: where mask[i] is true, set output[i] = fill_value;
/// otherwise output[i] = input[i].
pub fn masked_fill_f32(input: &[f32], mask: &[bool], fill_value: f32, output: &mut [f32]) {
    assert_eq!(input.len(), mask.len());
    assert_eq!(input.len(), output.len());
    for i in 0..input.len() {
        output[i] = if mask[i] { fill_value } else { input[i] };
    }
}

/// Checked version of `masked_fill_f32`.
pub fn masked_fill_f32_checked(
    input: &[f32],
    mask: &[bool],
    fill_value: f32,
    output: &mut [f32],
) -> nnx_core::error::Result<()> {
    if input.len() != mask.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "masked_fill: input.len()={} != mask.len()={}",
            input.len(),
            mask.len()
        )));
    }
    if input.len() != output.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "masked_fill: input.len()={} != output.len()={}",
            input.len(),
            output.len()
        )));
    }
    masked_fill_f32(input, mask, fill_value, output);
    Ok(())
}

/// Masked fill in-place.
pub fn masked_fill_f32_inplace(data: &mut [f32], mask: &[bool], fill_value: f32) {
    assert_eq!(data.len(), mask.len());
    for i in 0..data.len() {
        if mask[i] {
            data[i] = fill_value;
        }
    }
}

/// Checked version of `masked_fill_f32_inplace`.
pub fn masked_fill_f32_inplace_checked(
    data: &mut [f32],
    mask: &[bool],
    fill_value: f32,
) -> nnx_core::error::Result<()> {
    if data.len() != mask.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "masked_fill_inplace: data.len()={} != mask.len()={}",
            data.len(),
            mask.len()
        )));
    }
    masked_fill_f32_inplace(data, mask, fill_value);
    Ok(())
}

/// Create a causal attention mask: [seq_len, seq_len] where
/// mask[i][j] = true if j > i (positions that should be masked/filled with -inf).
///
/// This is the standard autoregressive mask used in decoder-only transformers.
pub fn causal_mask(output: &mut [bool], seq_len: usize) {
    assert_eq!(output.len(), seq_len * seq_len);
    triu_mask(output, seq_len, seq_len, 1);
}

/// Checked version of `causal_mask`.
pub fn causal_mask_checked(output: &mut [bool], seq_len: usize) -> nnx_core::error::Result<()> {
    if output.len() != seq_len * seq_len {
        return Err(EngineError::ShapeMismatch(format!(
            "causal_mask: output.len()={} but seq_len*seq_len={}",
            output.len(),
            seq_len * seq_len
        )));
    }
    causal_mask(output, seq_len);
    Ok(())
}

/// Apply causal mask to attention scores [seq_len, seq_len]:
/// scores[i][j] = -inf where j > i.
pub fn apply_causal_mask_f32(scores: &mut [f32], seq_len: usize) {
    assert_eq!(scores.len(), seq_len * seq_len);
    for r in 0..seq_len {
        for c in (r + 1)..seq_len {
            scores[r * seq_len + c] = f32::NEG_INFINITY;
        }
    }
}

/// Checked version of `apply_causal_mask_f32`.
pub fn apply_causal_mask_f32_checked(
    scores: &mut [f32],
    seq_len: usize,
) -> nnx_core::error::Result<()> {
    if scores.len() != seq_len * seq_len {
        return Err(EngineError::ShapeMismatch(format!(
            "apply_causal_mask: scores.len()={} but seq_len*seq_len={}",
            scores.len(),
            seq_len * seq_len
        )));
    }
    apply_causal_mask_f32(scores, seq_len);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triu_mask() {
        let mut mask = [false; 9];
        triu_mask(&mut mask, 3, 3, 0);
        assert_eq!(
            mask,
            [true, true, true, false, true, true, false, false, true]
        );
    }

    #[test]
    fn test_tril_mask() {
        let mut mask = [false; 9];
        tril_mask(&mut mask, 3, 3, 0);
        assert_eq!(
            mask,
            [true, false, false, true, true, false, true, true, true]
        );
    }

    #[test]
    fn test_triu_f32() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0f32];
        triu_f32(&mut data, 3, 3, 1, 0.0);
        assert_eq!(data, [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_masked_fill() {
        let input = [1.0, 2.0, 3.0, 4.0f32];
        let mask = [false, true, false, true];
        let mut output = [0.0f32; 4];
        masked_fill_f32(&input, &mask, -999.0, &mut output);
        assert_eq!(output, [1.0, -999.0, 3.0, -999.0]);
    }

    #[test]
    fn test_causal_mask() {
        let mut mask = [false; 9];
        causal_mask(&mut mask, 3);
        assert_eq!(
            mask,
            [false, true, true, false, false, true, false, false, false]
        );
    }

    #[test]
    fn test_apply_causal_mask() {
        let mut scores = [1.0f32; 9];
        apply_causal_mask_f32(&mut scores, 3);
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[3], 1.0);
        assert_eq!(scores[4], 1.0);
        assert_eq!(scores[5], f32::NEG_INFINITY);
        assert_eq!(scores[8], 1.0);
    }

    // Checked wrapper tests
    #[test]
    fn test_triu_mask_checked_valid() {
        let mut mask = [false; 9];
        assert!(triu_mask_checked(&mut mask, 3, 3, 0).is_ok());
    }

    #[test]
    fn test_triu_mask_checked_bad_size() {
        let mut mask = [false; 8]; // wrong size for 3x3
        assert!(triu_mask_checked(&mut mask, 3, 3, 0).is_err());
    }

    #[test]
    fn test_masked_fill_checked_valid() {
        let input = [1.0, 2.0f32];
        let mask = [false, true];
        let mut output = [0.0f32; 2];
        assert!(masked_fill_f32_checked(&input, &mask, -1.0, &mut output).is_ok());
        assert_eq!(output, [1.0, -1.0]);
    }

    #[test]
    fn test_masked_fill_checked_mismatch() {
        let input = [1.0, 2.0f32];
        let mask = [false, true, false]; // wrong size
        let mut output = [0.0f32; 2];
        assert!(masked_fill_f32_checked(&input, &mask, -1.0, &mut output).is_err());
    }

    #[test]
    fn test_causal_mask_checked_bad_size() {
        let mut mask = [false; 8]; // wrong for seq_len=3
        assert!(causal_mask_checked(&mut mask, 3).is_err());
    }

    #[test]
    fn test_apply_causal_mask_checked_bad_size() {
        let mut scores = [1.0f32; 8]; // wrong for seq_len=3
        assert!(apply_causal_mask_f32_checked(&mut scores, 3).is_err());
    }
}
