//! Tensor creation operations.
//!
//! Arange, Range, ConstantOfShape, Linspace, full, zeros, ones.

use nnx_core::error::EngineError;

/// Generate a range [start, start+step, start+2*step, ...) while < stop.
/// Returns the number of elements written to `output`.
pub fn arange_f32(output: &mut [f32], start: f32, stop: f32, step: f32) -> usize {
    assert!(step != 0.0, "step must be non-zero");
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    assert!(output.len() >= n, "output buffer too small: need {n}, got {}", output.len());
    for i in 0..n {
        output[i] = start + i as f32 * step;
    }
    n
}

/// Checked version of `arange_f32`.
pub fn arange_f32_checked(output: &mut [f32], start: f32, stop: f32, step: f32) -> nnx_core::error::Result<usize> {
    if step == 0.0 {
        return Err(EngineError::Kernel(
            "arange: step must be non-zero".to_string()
        ));
    }
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    if output.len() < n {
        return Err(EngineError::ShapeMismatch(
            format!("arange: output.len()={} but need at least {}", output.len(), n)
        ));
    }
    Ok(arange_f32(output, start, stop, step))
}

/// Compute the number of elements arange would produce.
pub fn arange_len(start: f32, stop: f32, step: f32) -> usize {
    assert!(step != 0.0, "step must be non-zero");
    ((stop - start) / step).ceil().max(0.0) as usize
}

/// Checked version of `arange_len`.
pub fn arange_len_checked(start: f32, stop: f32, step: f32) -> nnx_core::error::Result<usize> {
    if step == 0.0 {
        return Err(EngineError::Kernel(
            "arange_len: step must be non-zero".to_string()
        ));
    }
    Ok(((stop - start) / step).ceil().max(0.0) as usize)
}

/// Fill `output` with evenly-spaced values from `start` to `stop` (inclusive).
pub fn linspace_f32(output: &mut [f32], start: f32, stop: f32) {
    let n = output.len();
    if n == 0 { return; }
    if n == 1 {
        output[0] = start;
        return;
    }
    let step = (stop - start) / (n - 1) as f32;
    for i in 0..n {
        output[i] = start + i as f32 * step;
    }
}

/// Fill output with a constant value -- the ONNX ConstantOfShape op.
pub fn constant_of_shape_f32(output: &mut [f32], value: f32) {
    output.fill(value);
}

/// Fill with zeros.
pub fn zeros_f32(output: &mut [f32]) {
    output.fill(0.0);
}

/// Fill with ones.
pub fn ones_f32(output: &mut [f32]) {
    output.fill(1.0);
}

/// Fill with a scalar.
pub fn full_f32(output: &mut [f32], value: f32) {
    output.fill(value);
}

/// Create an identity matrix [n, n] (flat row-major).
pub fn eye_f32(output: &mut [f32], n: usize) {
    assert_eq!(output.len(), n * n);
    output.fill(0.0);
    for i in 0..n {
        output[i * n + i] = 1.0;
    }
}

/// Checked version of `eye_f32`.
pub fn eye_f32_checked(output: &mut [f32], n: usize) -> nnx_core::error::Result<()> {
    if output.len() != n * n {
        return Err(EngineError::ShapeMismatch(
            format!("eye: output.len()={} but n*n={}", output.len(), n * n)
        ));
    }
    eye_f32(output, n);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arange_basic() {
        let mut out = [0.0f32; 5];
        let n = arange_f32(&mut out, 0.0, 5.0, 1.0);
        assert_eq!(n, 5);
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arange_step() {
        let mut out = [0.0f32; 3];
        let n = arange_f32(&mut out, 1.0, 7.0, 2.0);
        assert_eq!(n, 3);
        assert_eq!(out, [1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_arange_negative() {
        let mut out = [0.0f32; 5];
        let n = arange_f32(&mut out, 4.0, -1.0, -1.0);
        assert_eq!(n, 5);
        assert_eq!(out, [4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_arange_len() {
        assert_eq!(arange_len(0.0, 5.0, 1.0), 5);
        assert_eq!(arange_len(0.0, 10.0, 3.0), 4);
        assert_eq!(arange_len(5.0, 0.0, 1.0), 0);
    }

    #[test]
    fn test_linspace() {
        let mut out = [0.0f32; 5];
        linspace_f32(&mut out, 0.0, 1.0);
        assert_eq!(out, [0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_constant_of_shape() {
        let mut out = [0.0f32; 4];
        constant_of_shape_f32(&mut out, 7.0);
        assert_eq!(out, [7.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_eye() {
        let mut out = [0.0f32; 9];
        eye_f32(&mut out, 3);
        assert_eq!(out, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    // Checked wrapper tests
    #[test]
    fn test_arange_checked_valid() {
        let mut out = [0.0f32; 5];
        let result = arange_f32_checked(&mut out, 0.0, 5.0, 1.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5);
    }

    #[test]
    fn test_arange_checked_zero_step() {
        let mut out = [0.0f32; 5];
        let result = arange_f32_checked(&mut out, 0.0, 5.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_arange_checked_buffer_too_small() {
        let mut out = [0.0f32; 2]; // too small for 0..5
        let result = arange_f32_checked(&mut out, 0.0, 5.0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_eye_checked_valid() {
        let mut out = [0.0f32; 9];
        assert!(eye_f32_checked(&mut out, 3).is_ok());
    }

    #[test]
    fn test_eye_checked_bad_size() {
        let mut out = [0.0f32; 8]; // wrong for n=3
        assert!(eye_f32_checked(&mut out, 3).is_err());
    }
}
