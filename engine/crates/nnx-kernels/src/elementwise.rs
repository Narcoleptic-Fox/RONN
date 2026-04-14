//! Element-wise math operations.
//!
//! All ops work on flat f32 slices. Broadcasting is handled at a higher level.

use nnx_core::error::EngineError;

/// Element-wise add: output = a + b
pub fn add_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] + b[i];
    }
}

/// Element-wise subtract: output = a - b
pub fn sub_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] - b[i];
    }
}

/// Element-wise multiply: output = a * b
pub fn mul_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] * b[i];
    }
}

/// Element-wise divide: output = a / b
pub fn div_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] / b[i];
    }
}

/// Negate: output = -x
pub fn neg_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = -x[i];
    }
}

/// Absolute value
pub fn abs_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].abs();
    }
}

/// Square root
pub fn sqrt_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].sqrt();
    }
}

/// Exponential
pub fn exp_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].exp();
    }
}

/// Natural log
pub fn log_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].ln();
    }
}

/// Power: output = base^exp (element-wise)
pub fn pow_f32(base: &[f32], exp: &[f32], output: &mut [f32]) {
    assert_eq!(base.len(), exp.len());
    assert_eq!(base.len(), output.len());
    for i in 0..base.len() {
        output[i] = base[i].powf(exp[i]);
    }
}

/// Scalar power: output = x^p
pub fn pow_scalar_f32(x: &[f32], p: f32, output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].powf(p);
    }
}

/// Floor
pub fn floor_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].floor();
    }
}

/// Ceil
pub fn ceil_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].ceil();
    }
}

/// Scalar add: output = x + s
pub fn add_scalar_f32(x: &[f32], s: f32, output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i] + s;
    }
}

/// Scalar multiply: output = x * s
pub fn mul_scalar_f32(x: &[f32], s: f32, output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i] * s;
    }
}

/// Where/Select: output[i] = if cond[i] { a[i] } else { b[i] }
pub fn where_f32(cond: &[bool], a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(cond.len(), a.len());
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = if cond[i] { a[i] } else { b[i] };
    }
}

// -- Checked wrappers --

/// Helper to validate binary element-wise op arguments.
fn validate_binary_elementwise(
    fn_name: &str,
    a: &[f32],
    b: &[f32],
    output: &[f32],
) -> nnx_core::error::Result<()> {
    if a.len() != b.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "{fn_name}: a.len()={} != b.len()={}",
            a.len(),
            b.len()
        )));
    }
    if a.len() != output.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "{fn_name}: a.len()={} != output.len()={}",
            a.len(),
            output.len()
        )));
    }
    Ok(())
}

/// Helper to validate unary element-wise op arguments.
fn validate_unary_elementwise(
    fn_name: &str,
    x: &[f32],
    output: &[f32],
) -> nnx_core::error::Result<()> {
    if x.len() != output.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "{fn_name}: x.len()={} != output.len()={}",
            x.len(),
            output.len()
        )));
    }
    Ok(())
}

/// Checked version of `add_f32`.
pub fn add_f32_checked(a: &[f32], b: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    validate_binary_elementwise("add", a, b, output)?;
    add_f32(a, b, output);
    Ok(())
}

/// Checked version of `sub_f32`.
pub fn sub_f32_checked(a: &[f32], b: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    validate_binary_elementwise("sub", a, b, output)?;
    sub_f32(a, b, output);
    Ok(())
}

/// Checked version of `mul_f32`.
pub fn mul_f32_checked(a: &[f32], b: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    validate_binary_elementwise("mul", a, b, output)?;
    mul_f32(a, b, output);
    Ok(())
}

/// Checked version of `div_f32`.
pub fn div_f32_checked(a: &[f32], b: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    validate_binary_elementwise("div", a, b, output)?;
    div_f32(a, b, output);
    Ok(())
}

/// Checked version of `neg_f32`.
pub fn neg_f32_checked(x: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    validate_unary_elementwise("neg", x, output)?;
    neg_f32(x, output);
    Ok(())
}

/// Checked version of `sqrt_f32`.
pub fn sqrt_f32_checked(x: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    validate_unary_elementwise("sqrt", x, output)?;
    sqrt_f32(x, output);
    Ok(())
}

/// Checked version of `exp_f32`.
pub fn exp_f32_checked(x: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    validate_unary_elementwise("exp", x, output)?;
    exp_f32(x, output);
    Ok(())
}

/// Checked version of `log_f32`.
pub fn log_f32_checked(x: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    validate_unary_elementwise("log", x, output)?;
    log_f32(x, output);
    Ok(())
}

/// Checked version of `where_f32`.
pub fn where_f32_checked(
    cond: &[bool],
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) -> nnx_core::error::Result<()> {
    if cond.len() != a.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "where: cond.len()={} != a.len()={}",
            cond.len(),
            a.len()
        )));
    }
    if a.len() != b.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "where: a.len()={} != b.len()={}",
            a.len(),
            b.len()
        )));
    }
    if a.len() != output.len() {
        return Err(EngineError::ShapeMismatch(format!(
            "where: a.len()={} != output.len()={}",
            a.len(),
            output.len()
        )));
    }
    where_f32(cond, a, b, output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [4.0, 5.0, 6.0f32];
        let mut out = [0.0f32; 3];

        add_f32(&a, &b, &mut out);
        assert_eq!(out, [5.0, 7.0, 9.0]);

        sub_f32(&a, &b, &mut out);
        assert_eq!(out, [-3.0, -3.0, -3.0]);

        mul_f32(&a, &b, &mut out);
        assert_eq!(out, [4.0, 10.0, 18.0]);

        div_f32(&a, &b, &mut out);
        for i in 0..3 {
            assert!((out[i] - a[i] / b[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_unary() {
        let x = [4.0, 9.0, 16.0f32];
        let mut out = [0.0f32; 3];
        sqrt_f32(&x, &mut out);
        assert_eq!(out, [2.0, 3.0, 4.0]);

        neg_f32(&x, &mut out);
        assert_eq!(out, [-4.0, -9.0, -16.0]);

        let neg_copy = out;
        abs_f32(&neg_copy, &mut out);
        assert_eq!(out, [4.0, 9.0, 16.0]);
    }

    // Checked wrapper tests
    #[test]
    fn test_add_checked_valid() {
        let a = [1.0, 2.0f32];
        let b = [3.0, 4.0f32];
        let mut out = [0.0f32; 2];
        assert!(add_f32_checked(&a, &b, &mut out).is_ok());
        assert_eq!(out, [4.0, 6.0]);
    }

    #[test]
    fn test_add_checked_mismatch() {
        let a = [1.0, 2.0f32];
        let b = [3.0f32]; // wrong size
        let mut out = [0.0f32; 2];
        assert!(add_f32_checked(&a, &b, &mut out).is_err());
    }

    #[test]
    fn test_neg_checked_mismatch() {
        let x = [1.0, 2.0f32];
        let mut out = [0.0f32; 3]; // wrong size
        assert!(neg_f32_checked(&x, &mut out).is_err());
    }

    #[test]
    fn test_where_checked_valid() {
        let cond = [true, false];
        let a = [10.0, 20.0f32];
        let b = [30.0, 40.0f32];
        let mut out = [0.0f32; 2];
        assert!(where_f32_checked(&cond, &a, &b, &mut out).is_ok());
        assert_eq!(out, [10.0, 40.0]);
    }

    #[test]
    fn test_where_checked_cond_mismatch() {
        let cond = [true];
        let a = [10.0, 20.0f32];
        let b = [30.0, 40.0f32];
        let mut out = [0.0f32; 2];
        assert!(where_f32_checked(&cond, &a, &b, &mut out).is_err());
    }
}
