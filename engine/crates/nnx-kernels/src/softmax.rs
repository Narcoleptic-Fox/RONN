//! Softmax kernel for attention score normalization.

/// In-place softmax over a slice.
pub fn softmax_f32(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // Numerical stability: subtract max
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_uniform() {
        let mut x = [1.0, 1.0, 1.0, 1.0f32];
        softmax_f32(&mut x);
        for &v in &x {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0f32];
        softmax_f32(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
