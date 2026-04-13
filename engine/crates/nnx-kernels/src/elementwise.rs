//! Element-wise math operations.
//!
//! All ops work on flat f32 slices. Broadcasting is handled at a higher level.

/// Element-wise add: output = a + b
pub fn add_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() { output[i] = a[i] + b[i]; }
}

/// Element-wise subtract: output = a - b
pub fn sub_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() { output[i] = a[i] - b[i]; }
}

/// Element-wise multiply: output = a * b
pub fn mul_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() { output[i] = a[i] * b[i]; }
}

/// Element-wise divide: output = a / b
pub fn div_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() { output[i] = a[i] / b[i]; }
}

/// Negate: output = -x
pub fn neg_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = -x[i]; }
}

/// Absolute value
pub fn abs_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i].abs(); }
}

/// Square root
pub fn sqrt_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i].sqrt(); }
}

/// Exponential
pub fn exp_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i].exp(); }
}

/// Natural log
pub fn log_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i].ln(); }
}

/// Power: output = base^exp (element-wise)
pub fn pow_f32(base: &[f32], exp: &[f32], output: &mut [f32]) {
    assert_eq!(base.len(), exp.len());
    assert_eq!(base.len(), output.len());
    for i in 0..base.len() { output[i] = base[i].powf(exp[i]); }
}

/// Scalar power: output = x^p
pub fn pow_scalar_f32(x: &[f32], p: f32, output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i].powf(p); }
}

/// Floor
pub fn floor_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i].floor(); }
}

/// Ceil
pub fn ceil_f32(x: &[f32], output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i].ceil(); }
}

/// Scalar add: output = x + s
pub fn add_scalar_f32(x: &[f32], s: f32, output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i] + s; }
}

/// Scalar multiply: output = x * s
pub fn mul_scalar_f32(x: &[f32], s: f32, output: &mut [f32]) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() { output[i] = x[i] * s; }
}

/// Where/Select: output[i] = if cond[i] { a[i] } else { b[i] }
pub fn where_f32(cond: &[bool], a: &[f32], b: &[f32], output: &mut [f32]) {
    assert_eq!(cond.len(), a.len());
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), output.len());
    for i in 0..a.len() { output[i] = if cond[i] { a[i] } else { b[i] }; }
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
        for i in 0..3 { assert!((out[i] - a[i] / b[i]).abs() < 1e-6); }
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
}
