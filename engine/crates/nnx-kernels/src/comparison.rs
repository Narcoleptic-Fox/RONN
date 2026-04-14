//! Comparison operations.
//!
//! Return `bool` slices for use with `where_f32` / `masked_fill`.

use nnx_core::error::EngineError;

/// Element-wise equal: output[i] = (a[i] == b[i])
pub fn equal_f32(a: &[f32], b: &[f32], output: &mut [bool]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = (a[i] - b[i]).abs() < f32::EPSILON;
    }
}

/// Element-wise greater: output[i] = (a[i] > b[i])
pub fn greater_f32(a: &[f32], b: &[f32], output: &mut [bool]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] > b[i];
    }
}

/// Element-wise less: output[i] = (a[i] < b[i])
pub fn less_f32(a: &[f32], b: &[f32], output: &mut [bool]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] < b[i];
    }
}

/// Element-wise greater-or-equal: output[i] = (a[i] >= b[i])
pub fn greater_or_equal_f32(a: &[f32], b: &[f32], output: &mut [bool]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] >= b[i];
    }
}

/// Element-wise less-or-equal: output[i] = (a[i] <= b[i])
pub fn less_or_equal_f32(a: &[f32], b: &[f32], output: &mut [bool]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] <= b[i];
    }
}

/// Logical NOT: output[i] = !input[i]
pub fn not_bool(input: &[bool], output: &mut [bool]) {
    assert_eq!(input.len(), output.len());
    for i in 0..input.len() {
        output[i] = !input[i];
    }
}

/// Logical AND: output[i] = a[i] && b[i]
pub fn and_bool(a: &[bool], b: &[bool], output: &mut [bool]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] && b[i];
    }
}

/// Logical OR: output[i] = a[i] || b[i]
pub fn or_bool(a: &[bool], b: &[bool], output: &mut [bool]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = a[i] || b[i];
    }
}

/// Compare against scalar: output[i] = (x[i] > scalar)
pub fn greater_scalar_f32(x: &[f32], scalar: f32, output: &mut [bool]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i] > scalar;
    }
}

/// Compare against scalar: output[i] = (x[i] < scalar)
pub fn less_scalar_f32(x: &[f32], scalar: f32, output: &mut [bool]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i] < scalar;
    }
}

/// Element-wise not-equal: output[i] = (a[i] != b[i])
pub fn not_equal_f32(a: &[f32], b: &[f32], output: &mut [bool]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() {
        output[i] = (a[i] - b[i]).abs() >= f32::EPSILON;
    }
}

// -- Checked wrappers for binary comparison ops --

/// Helper to validate binary comparison arguments.
fn validate_binary_f32(
    fn_name: &str,
    a: &[f32],
    b: &[f32],
    output: &[bool],
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

/// Checked version of `equal_f32`.
pub fn equal_f32_checked(a: &[f32], b: &[f32], output: &mut [bool]) -> nnx_core::error::Result<()> {
    validate_binary_f32("equal", a, b, output)?;
    equal_f32(a, b, output);
    Ok(())
}

/// Checked version of `greater_f32`.
pub fn greater_f32_checked(
    a: &[f32],
    b: &[f32],
    output: &mut [bool],
) -> nnx_core::error::Result<()> {
    validate_binary_f32("greater", a, b, output)?;
    greater_f32(a, b, output);
    Ok(())
}

/// Checked version of `less_f32`.
pub fn less_f32_checked(a: &[f32], b: &[f32], output: &mut [bool]) -> nnx_core::error::Result<()> {
    validate_binary_f32("less", a, b, output)?;
    less_f32(a, b, output);
    Ok(())
}

/// Checked version of `not_equal_f32`.
pub fn not_equal_f32_checked(
    a: &[f32],
    b: &[f32],
    output: &mut [bool],
) -> nnx_core::error::Result<()> {
    validate_binary_f32("not_equal", a, b, output)?;
    not_equal_f32(a, b, output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [1.0, 2.5, 3.0f32];
        let mut out = [false; 3];
        equal_f32(&a, &b, &mut out);
        assert_eq!(out, [true, false, true]);
    }

    #[test]
    fn test_greater_less() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [2.0, 2.0, 1.0f32];
        let mut gt = [false; 3];
        let mut lt = [false; 3];
        greater_f32(&a, &b, &mut gt);
        less_f32(&a, &b, &mut lt);
        assert_eq!(gt, [false, false, true]);
        assert_eq!(lt, [true, false, false]);
    }

    #[test]
    fn test_not() {
        let input = [true, false, true];
        let mut out = [false; 3];
        not_bool(&input, &mut out);
        assert_eq!(out, [false, true, false]);
    }

    #[test]
    fn test_logical_ops() {
        let a = [true, true, false, false];
        let b = [true, false, true, false];
        let mut and_out = [false; 4];
        let mut or_out = [false; 4];
        and_bool(&a, &b, &mut and_out);
        or_bool(&a, &b, &mut or_out);
        assert_eq!(and_out, [true, false, false, false]);
        assert_eq!(or_out, [true, true, true, false]);
    }

    #[test]
    fn test_scalar_comparison() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0f32];
        let mut gt = [false; 5];
        greater_scalar_f32(&x, 3.0, &mut gt);
        assert_eq!(gt, [false, false, false, true, true]);
    }

    #[test]
    fn test_greater_or_equal_less_or_equal() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [2.0, 2.0, 1.0f32];
        let mut ge = [false; 3];
        let mut le = [false; 3];
        greater_or_equal_f32(&a, &b, &mut ge);
        less_or_equal_f32(&a, &b, &mut le);
        assert_eq!(ge, [false, true, true]);
        assert_eq!(le, [true, true, false]);
    }

    // Checked wrapper tests
    #[test]
    fn test_equal_checked_valid() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [1.0, 2.5, 3.0f32];
        let mut out = [false; 3];
        assert!(equal_f32_checked(&a, &b, &mut out).is_ok());
        assert_eq!(out, [true, false, true]);
    }

    #[test]
    fn test_equal_checked_size_mismatch() {
        let a = [1.0, 2.0f32];
        let b = [1.0, 2.0, 3.0f32];
        let mut out = [false; 3];
        assert!(equal_f32_checked(&a, &b, &mut out).is_err());
    }

    #[test]
    fn test_greater_checked_output_mismatch() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [1.0, 2.0, 3.0f32];
        let mut out = [false; 2]; // wrong size
        assert!(greater_f32_checked(&a, &b, &mut out).is_err());
    }
}
