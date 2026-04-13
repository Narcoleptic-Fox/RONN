//! Reduction operations: Sum, Mean, Max, Min, ArgMax, ArgMin, Prod.
//!
//! These operate on flat slices. Axis-aware reduction is handled at
//! a higher level by computing strides and calling these on sub-slices.

use nnx_core::error::EngineError;

/// Sum of all elements.
pub fn sum_f32(x: &[f32]) -> f32 {
    x.iter().sum()
}

/// Mean of all elements.
pub fn mean_f32(x: &[f32]) -> f32 {
    if x.is_empty() { return 0.0; }
    x.iter().sum::<f32>() / x.len() as f32
}

/// Maximum value.
pub fn max_f32(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

/// Minimum value.
pub fn min_f32(x: &[f32]) -> f32 {
    x.iter().cloned().fold(f32::INFINITY, f32::min)
}

/// Index of maximum value.
pub fn argmax_f32(x: &[f32]) -> usize {
    x.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Index of minimum value.
pub fn argmin_f32(x: &[f32]) -> usize {
    x.iter().enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Product of all elements.
pub fn prod_f32(x: &[f32]) -> f32 {
    x.iter().product()
}

/// Variance of all elements.
pub fn var_f32(x: &[f32]) -> f32 {
    if x.len() < 2 { return 0.0; }
    let mean = mean_f32(x);
    x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32
}

/// Standard deviation.
pub fn std_f32(x: &[f32]) -> f32 {
    var_f32(x).sqrt()
}

/// Sum along axis for a 2D matrix [rows, cols].
/// axis=0: sum each column → [cols]
/// axis=1: sum each row → [rows]
pub fn sum_axis_2d(x: &[f32], output: &mut [f32], rows: usize, cols: usize, axis: usize) {
    match axis {
        0 => {
            assert_eq!(output.len(), cols);
            output.fill(0.0);
            for r in 0..rows {
                for c in 0..cols {
                    output[c] += x[r * cols + c];
                }
            }
        }
        1 => {
            assert_eq!(output.len(), rows);
            for r in 0..rows {
                output[r] = x[r * cols..(r + 1) * cols].iter().sum();
            }
        }
        _ => panic!("axis must be 0 or 1 for 2D"),
    }
}

/// Mean along axis for 2D.
pub fn mean_axis_2d(x: &[f32], output: &mut [f32], rows: usize, cols: usize, axis: usize) {
    sum_axis_2d(x, output, rows, cols, axis);
    let divisor = if axis == 0 { rows as f32 } else { cols as f32 };
    for v in output.iter_mut() { *v /= divisor; }
}

/// Checked version of `sum_axis_2d` that returns Result instead of panicking on bad axis.
pub fn sum_axis_2d_checked(
    x: &[f32], output: &mut [f32], rows: usize, cols: usize, axis: usize,
) -> nnx_core::error::Result<()> {
    if axis > 1 {
        return Err(EngineError::Kernel(
            format!("sum_axis_2d: axis must be 0 or 1, got {}", axis)
        ));
    }
    if x.len() != rows * cols {
        return Err(EngineError::ShapeMismatch(
            format!("sum_axis_2d: x.len()={} but rows*cols={}", x.len(), rows * cols)
        ));
    }
    let expected_out = if axis == 0 { cols } else { rows };
    if output.len() != expected_out {
        return Err(EngineError::ShapeMismatch(
            format!("sum_axis_2d: output.len()={} but expected {}", output.len(), expected_out)
        ));
    }
    sum_axis_2d(x, output, rows, cols, axis);
    Ok(())
}

/// Checked version of `mean_axis_2d`.
pub fn mean_axis_2d_checked(
    x: &[f32], output: &mut [f32], rows: usize, cols: usize, axis: usize,
) -> nnx_core::error::Result<()> {
    if axis > 1 {
        return Err(EngineError::Kernel(
            format!("mean_axis_2d: axis must be 0 or 1, got {}", axis)
        ));
    }
    if x.len() != rows * cols {
        return Err(EngineError::ShapeMismatch(
            format!("mean_axis_2d: x.len()={} but rows*cols={}", x.len(), rows * cols)
        ));
    }
    let expected_out = if axis == 0 { cols } else { rows };
    if output.len() != expected_out {
        return Err(EngineError::ShapeMismatch(
            format!("mean_axis_2d: output.len()={} but expected {}", output.len(), expected_out)
        ));
    }
    mean_axis_2d(x, output, rows, cols, axis);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_reductions() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0f32];
        assert_eq!(sum_f32(&x), 15.0);
        assert_eq!(mean_f32(&x), 3.0);
        assert_eq!(max_f32(&x), 5.0);
        assert_eq!(min_f32(&x), 1.0);
        assert_eq!(argmax_f32(&x), 4);
        assert_eq!(argmin_f32(&x), 0);
        assert_eq!(prod_f32(&x), 120.0);
    }

    #[test]
    fn test_sum_axis() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32]; // 2x3
        let mut out_cols = [0.0f32; 3];
        sum_axis_2d(&x, &mut out_cols, 2, 3, 0);
        assert_eq!(out_cols, [5.0, 7.0, 9.0]); // col sums

        let mut out_rows = [0.0f32; 2];
        sum_axis_2d(&x, &mut out_rows, 2, 3, 1);
        assert_eq!(out_rows, [6.0, 15.0]); // row sums
    }

    // Checked wrapper tests
    #[test]
    fn test_sum_axis_checked_valid() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let mut out = [0.0f32; 3];
        assert!(sum_axis_2d_checked(&x, &mut out, 2, 3, 0).is_ok());
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_axis_checked_bad_axis() {
        let x = [1.0, 2.0f32];
        let mut out = [0.0f32; 1];
        assert!(sum_axis_2d_checked(&x, &mut out, 1, 2, 2).is_err());
    }

    #[test]
    fn test_sum_axis_checked_bad_output() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let mut out = [0.0f32; 2]; // wrong for axis=0, cols=3
        assert!(sum_axis_2d_checked(&x, &mut out, 2, 3, 0).is_err());
    }
}
