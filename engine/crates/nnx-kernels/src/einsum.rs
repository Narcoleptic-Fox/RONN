//! Einsum -- Einstein summation convention.
//!
//! Supports common 2-operand patterns used in neural networks.
//! For arbitrary subscripts, we parse and dispatch to specialized kernels.

use crate::matmul::{dot_f32, matmul_f32};

use nnx_core::error::EngineError;

/// Supported einsum patterns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EinsumPattern {
    /// "ij,jk->ik" -- standard matrix multiply
    MatMul,
    /// "ij,j->i" -- matrix-vector multiply
    MatVec,
    /// "ij,ij->ij" -- element-wise (Hadamard) product
    Hadamard,
    /// "ij,ij->" -- Frobenius inner product (sum of element-wise product)
    FrobeniusInner,
    /// "ij->ji" -- transpose
    Transpose,
    /// "ij->i" -- row sum
    RowSum,
    /// "ij->j" -- column sum
    ColSum,
    /// "ij->" -- total sum
    TotalSum,
    /// "i,i->" -- dot product
    DotProduct,
    /// "i,j->ij" -- outer product
    OuterProduct,
    /// "bij,bjk->bik" -- batched matrix multiply
    BatchMatMul,
    /// "bhqd,bhkd->bhqk" -- attention scores (batched)
    AttentionScores,
    /// "bhqk,bhkd->bhqd" -- attention output (batched)
    AttentionOutput,
}

/// Parse an einsum subscript string into a pattern.
/// Returns None for unsupported patterns.
pub fn parse_einsum(subscripts: &str) -> Option<EinsumPattern> {
    let trimmed = subscripts.replace(' ', "");
    match trimmed.as_str() {
        "ij,jk->ik" => Some(EinsumPattern::MatMul),
        "ij,j->i" => Some(EinsumPattern::MatVec),
        "ij,ij->ij" => Some(EinsumPattern::Hadamard),
        "ij,ij->" => Some(EinsumPattern::FrobeniusInner),
        "ij->ji" => Some(EinsumPattern::Transpose),
        "ij->i" => Some(EinsumPattern::RowSum),
        "ij->j" => Some(EinsumPattern::ColSum),
        "ij->" => Some(EinsumPattern::TotalSum),
        "i,i->" => Some(EinsumPattern::DotProduct),
        "i,j->ij" => Some(EinsumPattern::OuterProduct),
        "bij,bjk->bik" => Some(EinsumPattern::BatchMatMul),
        "bhqd,bhkd->bhqk" => Some(EinsumPattern::AttentionScores),
        "bhqk,bhkd->bhqd" => Some(EinsumPattern::AttentionOutput),
        _ => None,
    }
}

/// Execute an einsum operation on two operands.
///
/// `a`, `b`: input tensors (flat f32)
/// `output`: result tensor (flat f32)
/// `a_shape`, `b_shape`: dimensions of inputs
///
/// Panics if the pattern is unsupported or shapes don't match.
pub fn einsum_f32(
    subscripts: &str,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    a_shape: &[usize],
    b_shape: &[usize],
) {
    let pattern = parse_einsum(subscripts)
        .unwrap_or_else(|| panic!("unsupported einsum pattern: {subscripts}"));

    match pattern {
        EinsumPattern::MatMul => {
            assert_eq!(a_shape.len(), 2);
            assert_eq!(b_shape.len(), 2);
            let (m, k) = (a_shape[0], a_shape[1]);
            let n = b_shape[1];
            assert_eq!(k, b_shape[0]);
            matmul_f32(a, b, output, m, k, n);
        }
        EinsumPattern::MatVec => {
            assert_eq!(a_shape.len(), 2);
            assert_eq!(b_shape.len(), 1);
            let (m, k) = (a_shape[0], a_shape[1]);
            assert_eq!(k, b_shape[0]);
            assert_eq!(output.len(), m);
            for i in 0..m {
                output[i] = dot_f32(&a[i * k..(i + 1) * k], b);
            }
        }
        EinsumPattern::Hadamard => {
            assert_eq!(a.len(), b.len());
            assert_eq!(a.len(), output.len());
            for i in 0..a.len() {
                output[i] = a[i] * b[i];
            }
        }
        EinsumPattern::FrobeniusInner => {
            assert_eq!(a.len(), b.len());
            assert_eq!(output.len(), 1);
            output[0] = dot_f32(a, b);
        }
        EinsumPattern::DotProduct => {
            assert_eq!(a.len(), b.len());
            assert_eq!(output.len(), 1);
            output[0] = dot_f32(a, b);
        }
        EinsumPattern::OuterProduct => {
            let m = a_shape[0];
            let n = b_shape[0];
            assert_eq!(output.len(), m * n);
            for i in 0..m {
                for j in 0..n {
                    output[i * n + j] = a[i] * b[j];
                }
            }
        }
        EinsumPattern::BatchMatMul => {
            assert_eq!(a_shape.len(), 3);
            assert_eq!(b_shape.len(), 3);
            let batch = a_shape[0];
            let (m, k) = (a_shape[1], a_shape[2]);
            let n = b_shape[2];
            assert_eq!(batch, b_shape[0]);
            assert_eq!(k, b_shape[1]);

            let a_batch_stride = m * k;
            let b_batch_stride = k * n;
            let c_batch_stride = m * n;

            for bi in 0..batch {
                matmul_f32(
                    &a[bi * a_batch_stride..(bi + 1) * a_batch_stride],
                    &b[bi * b_batch_stride..(bi + 1) * b_batch_stride],
                    &mut output[bi * c_batch_stride..(bi + 1) * c_batch_stride],
                    m,
                    k,
                    n,
                );
            }
        }
        EinsumPattern::AttentionScores => {
            // bhqd,bhkd->bhqk: for each (b,h), Q[q,d] @ K[k,d]^T -> scores[q,k]
            assert_eq!(a_shape.len(), 4);
            assert_eq!(b_shape.len(), 4);
            let (b_size, h_size, q_len, d_dim) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
            let k_len = b_shape[2];
            assert_eq!(d_dim, b_shape[3]);

            for bi in 0..b_size {
                for hi in 0..h_size {
                    let q_offset = ((bi * h_size + hi) * q_len) * d_dim;
                    let k_offset = ((bi * h_size + hi) * k_len) * d_dim;
                    let o_offset = ((bi * h_size + hi) * q_len) * k_len;

                    for qi in 0..q_len {
                        for ki in 0..k_len {
                            let q_row = &a[q_offset + qi * d_dim..q_offset + (qi + 1) * d_dim];
                            let k_row = &b[k_offset + ki * d_dim..k_offset + (ki + 1) * d_dim];
                            output[o_offset + qi * k_len + ki] = dot_f32(q_row, k_row);
                        }
                    }
                }
            }
        }
        EinsumPattern::AttentionOutput => {
            // bhqk,bhkd->bhqd: for each (b,h), scores[q,k] @ V[k,d] -> out[q,d]
            assert_eq!(a_shape.len(), 4);
            assert_eq!(b_shape.len(), 4);
            let (b_size, h_size, q_len, k_len) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
            let d_dim = b_shape[3];
            assert_eq!(k_len, b_shape[2]);

            for bi in 0..b_size {
                for hi in 0..h_size {
                    let a_offset = ((bi * h_size + hi) * q_len) * k_len;
                    let b_offset = ((bi * h_size + hi) * k_len) * d_dim;
                    let o_offset = ((bi * h_size + hi) * q_len) * d_dim;

                    matmul_f32(
                        &a[a_offset..a_offset + q_len * k_len],
                        &b[b_offset..b_offset + k_len * d_dim],
                        &mut output[o_offset..o_offset + q_len * d_dim],
                        q_len,
                        k_len,
                        d_dim,
                    );
                }
            }
        }
        EinsumPattern::Transpose
        | EinsumPattern::RowSum
        | EinsumPattern::ColSum
        | EinsumPattern::TotalSum => {
            // These are single-operand -- use einsum_unary_f32 instead
            panic!("use einsum_unary_f32 for single-operand patterns like {subscripts}");
        }
    }
}

/// Checked version of `einsum_f32` that returns `Result` instead of panicking.
pub fn einsum_f32_checked(
    subscripts: &str,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    a_shape: &[usize],
    b_shape: &[usize],
) -> nnx_core::error::Result<()> {
    let pattern = parse_einsum(subscripts)
        .ok_or_else(|| EngineError::Kernel(format!("unsupported einsum pattern: {subscripts}")))?;

    match pattern {
        EinsumPattern::Transpose
        | EinsumPattern::RowSum
        | EinsumPattern::ColSum
        | EinsumPattern::TotalSum => {
            return Err(EngineError::Kernel(format!(
                "use einsum_unary_f32_checked for single-operand patterns like {subscripts}"
            )));
        }
        EinsumPattern::MatMul => {
            if a_shape.len() != 2 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatMul: a_shape must be 2D, got {}D",
                    a_shape.len()
                )));
            }
            if b_shape.len() != 2 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatMul: b_shape must be 2D, got {}D",
                    b_shape.len()
                )));
            }
            let (m, k) = (a_shape[0], a_shape[1]);
            let n = b_shape[1];
            if k != b_shape[0] {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatMul: a_shape[1]={} != b_shape[0]={}",
                    k, b_shape[0]
                )));
            }
            if a.len() != m * k {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatMul: a.len()={} but m*k={}",
                    a.len(),
                    m * k
                )));
            }
            if b.len() != k * n {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatMul: b.len()={} but k*n={}",
                    b.len(),
                    k * n
                )));
            }
            if output.len() != m * n {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatMul: output.len()={} but m*n={}",
                    output.len(),
                    m * n
                )));
            }
        }
        EinsumPattern::MatVec => {
            if a_shape.len() != 2 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatVec: a_shape must be 2D, got {}D",
                    a_shape.len()
                )));
            }
            if b_shape.len() != 1 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatVec: b_shape must be 1D, got {}D",
                    b_shape.len()
                )));
            }
            let (m, k) = (a_shape[0], a_shape[1]);
            if k != b_shape[0] {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatVec: a_shape[1]={} != b_shape[0]={}",
                    k, b_shape[0]
                )));
            }
            if output.len() != m {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum MatVec: output.len()={} but m={}",
                    output.len(),
                    m
                )));
            }
        }
        EinsumPattern::Hadamard => {
            if a.len() != b.len() || a.len() != output.len() {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum Hadamard: a.len()={}, b.len()={}, output.len()={} must all match",
                    a.len(),
                    b.len(),
                    output.len()
                )));
            }
        }
        EinsumPattern::FrobeniusInner | EinsumPattern::DotProduct => {
            if a.len() != b.len() {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum dot/frobenius: a.len()={} != b.len()={}",
                    a.len(),
                    b.len()
                )));
            }
            if output.len() != 1 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum dot/frobenius: output.len()={} but expected 1",
                    output.len()
                )));
            }
        }
        EinsumPattern::OuterProduct => {
            let m = a_shape[0];
            let n = b_shape[0];
            if output.len() != m * n {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum OuterProduct: output.len()={} but m*n={}",
                    output.len(),
                    m * n
                )));
            }
        }
        EinsumPattern::BatchMatMul => {
            if a_shape.len() != 3 || b_shape.len() != 3 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum BatchMatMul: shapes must be 3D, got a={}D b={}D",
                    a_shape.len(),
                    b_shape.len()
                )));
            }
            if a_shape[0] != b_shape[0] {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum BatchMatMul: batch dims differ a[0]={} b[0]={}",
                    a_shape[0], b_shape[0]
                )));
            }
            if a_shape[2] != b_shape[1] {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum BatchMatMul: inner dims differ a[2]={} b[1]={}",
                    a_shape[2], b_shape[1]
                )));
            }
        }
        EinsumPattern::AttentionScores | EinsumPattern::AttentionOutput => {
            if a_shape.len() != 4 || b_shape.len() != 4 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum attention: shapes must be 4D, got a={}D b={}D",
                    a_shape.len(),
                    b_shape.len()
                )));
            }
        }
    }

    // All validations passed; delegate to the unchecked version.
    einsum_f32(subscripts, a, b, output, a_shape, b_shape);
    Ok(())
}

/// Single-operand einsum (transpose, reductions).
pub fn einsum_unary_f32(subscripts: &str, input: &[f32], output: &mut [f32], shape: &[usize]) {
    let pattern = parse_einsum(subscripts)
        .unwrap_or_else(|| panic!("unsupported einsum pattern: {subscripts}"));

    match pattern {
        EinsumPattern::Transpose => {
            assert_eq!(shape.len(), 2);
            let (rows, cols) = (shape[0], shape[1]);
            crate::shape_ops::transpose_2d_f32(input, output, rows, cols);
        }
        EinsumPattern::RowSum => {
            assert_eq!(shape.len(), 2);
            let (rows, cols) = (shape[0], shape[1]);
            assert_eq!(output.len(), rows);
            for r in 0..rows {
                output[r] = input[r * cols..(r + 1) * cols].iter().sum();
            }
        }
        EinsumPattern::ColSum => {
            assert_eq!(shape.len(), 2);
            let (rows, cols) = (shape[0], shape[1]);
            assert_eq!(output.len(), cols);
            output.fill(0.0);
            for r in 0..rows {
                for c in 0..cols {
                    output[c] += input[r * cols + c];
                }
            }
        }
        EinsumPattern::TotalSum => {
            assert_eq!(output.len(), 1);
            output[0] = input.iter().sum();
        }
        _ => panic!("pattern requires two operands: {subscripts}"),
    }
}

/// Checked version of `einsum_unary_f32` that returns `Result` instead of panicking.
pub fn einsum_unary_f32_checked(
    subscripts: &str,
    input: &[f32],
    output: &mut [f32],
    shape: &[usize],
) -> nnx_core::error::Result<()> {
    let pattern = parse_einsum(subscripts)
        .ok_or_else(|| EngineError::Kernel(format!("unsupported einsum pattern: {subscripts}")))?;

    match pattern {
        EinsumPattern::Transpose => {
            if shape.len() != 2 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum Transpose: shape must be 2D, got {}D",
                    shape.len()
                )));
            }
            let (rows, cols) = (shape[0], shape[1]);
            if input.len() != rows * cols {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum Transpose: input.len()={} but rows*cols={}",
                    input.len(),
                    rows * cols
                )));
            }
            if output.len() != rows * cols {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum Transpose: output.len()={} but rows*cols={}",
                    output.len(),
                    rows * cols
                )));
            }
        }
        EinsumPattern::RowSum => {
            if shape.len() != 2 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum RowSum: shape must be 2D, got {}D",
                    shape.len()
                )));
            }
            let (rows, _cols) = (shape[0], shape[1]);
            if output.len() != rows {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum RowSum: output.len()={} but rows={}",
                    output.len(),
                    rows
                )));
            }
        }
        EinsumPattern::ColSum => {
            if shape.len() != 2 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum ColSum: shape must be 2D, got {}D",
                    shape.len()
                )));
            }
            let (_rows, cols) = (shape[0], shape[1]);
            if output.len() != cols {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum ColSum: output.len()={} but cols={}",
                    output.len(),
                    cols
                )));
            }
        }
        EinsumPattern::TotalSum => {
            if output.len() != 1 {
                return Err(EngineError::ShapeMismatch(format!(
                    "einsum TotalSum: output.len()={} but expected 1",
                    output.len()
                )));
            }
        }
        _ => {
            return Err(EngineError::Kernel(format!(
                "einsum_unary: pattern requires two operands: {subscripts}"
            )));
        }
    }

    einsum_unary_f32(subscripts, input, output, shape);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_patterns() {
        assert_eq!(parse_einsum("ij,jk->ik"), Some(EinsumPattern::MatMul));
        assert_eq!(parse_einsum("i,i->"), Some(EinsumPattern::DotProduct));
        assert_eq!(
            parse_einsum("bij,bjk->bik"),
            Some(EinsumPattern::BatchMatMul)
        );
        assert_eq!(parse_einsum("xyz"), None);
    }

    #[test]
    fn test_einsum_matmul() {
        let a = [1.0, 2.0, 3.0, 4.0f32]; // 2x2
        let b = [5.0, 6.0, 7.0, 8.0f32]; // 2x2
        let mut out = [0.0f32; 4];
        einsum_f32("ij,jk->ik", &a, &b, &mut out, &[2, 2], &[2, 2]);
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_einsum_dot_product() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [4.0, 5.0, 6.0f32];
        let mut out = [0.0f32; 1];
        einsum_f32("i,i->", &a, &b, &mut out, &[3], &[3]);
        assert!((out[0] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_einsum_outer_product() {
        let a = [1.0, 2.0f32];
        let b = [3.0, 4.0, 5.0f32];
        let mut out = [0.0f32; 6];
        einsum_f32("i,j->ij", &a, &b, &mut out, &[2], &[3]);
        assert_eq!(out, [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_einsum_batch_matmul() {
        let a = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0f32];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let mut out = [0.0f32; 8];
        einsum_f32("bij,bjk->bik", &a, &b, &mut out, &[2, 2, 2], &[2, 2, 2]);
        assert_eq!(out, b);
    }

    #[test]
    fn test_einsum_transpose() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // 2x3
        let mut out = [0.0f32; 6];
        einsum_unary_f32("ij->ji", &input, &mut out, &[2, 3]);
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_einsum_row_sum() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // 2x3
        let mut out = [0.0f32; 2];
        einsum_unary_f32("ij->i", &input, &mut out, &[2, 3]);
        assert_eq!(out, [6.0, 15.0]);
    }

    #[test]
    fn test_einsum_attention_scores() {
        let q = [1.0, 0.0, 0.0, 1.0f32];
        let k = [1.0, 0.0, 0.0, 1.0f32];
        let mut scores = [0.0f32; 4];
        einsum_f32(
            "bhqd,bhkd->bhqk",
            &q,
            &k,
            &mut scores,
            &[1, 1, 2, 2],
            &[1, 1, 2, 2],
        );
        assert_eq!(scores, [1.0, 0.0, 0.0, 1.0]);
    }

    // Checked wrapper tests
    #[test]
    fn test_einsum_checked_valid_matmul() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let mut out = [0.0f32; 4];
        let result = einsum_f32_checked("ij,jk->ik", &a, &b, &mut out, &[2, 2], &[2, 2]);
        assert!(result.is_ok());
        assert_eq!(out, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_einsum_checked_unsupported_pattern() {
        let a = [1.0f32];
        let b = [1.0f32];
        let mut out = [0.0f32; 1];
        let result = einsum_f32_checked("xyz", &a, &b, &mut out, &[1], &[1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_checked_shape_mismatch() {
        let a = [1.0, 2.0, 3.0, 4.0f32]; // 2x2
        let b = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0f32]; // 3x2
        let mut out = [0.0f32; 4];
        // a[1] = 2, b[0] = 3, mismatch
        let result = einsum_f32_checked("ij,jk->ik", &a, &b, &mut out, &[2, 2], &[3, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_checked_unary_as_binary() {
        let a = [1.0, 2.0f32];
        let b = [3.0, 4.0f32];
        let mut out = [0.0f32; 4];
        // Transpose is a unary op, should error in binary checked
        let result = einsum_f32_checked("ij->ji", &a, &b, &mut out, &[1, 2], &[1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_unary_checked_valid() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let mut out = [0.0f32; 6];
        let result = einsum_unary_f32_checked("ij->ji", &input, &mut out, &[2, 3]);
        assert!(result.is_ok());
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_einsum_unary_checked_unsupported() {
        let input = [1.0f32];
        let mut out = [0.0f32; 1];
        let result = einsum_unary_f32_checked("xyz", &input, &mut out, &[1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_unary_checked_wrong_shape() {
        let input = [1.0, 2.0, 3.0f32];
        let mut out = [0.0f32; 3]; // wrong: should be 2 for row sum of 2x...
        // RowSum with shape [2,3] => output should be [2]
        let result = einsum_unary_f32_checked("ij->i", &input, &mut out, &[2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_einsum_unary_checked_binary_pattern() {
        let input = [1.0, 2.0f32];
        let mut out = [0.0f32; 1];
        // MatMul is binary, should error in unary checked
        let result = einsum_unary_f32_checked("ij,jk->ik", &input, &mut out, &[1, 2]);
        assert!(result.is_err());
    }
}
