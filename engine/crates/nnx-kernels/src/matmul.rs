//! Matrix multiplication kernels.

/// Naive matrix multiply: C = A × B
/// A: [M, K], B: [K, N], C: [M, N]
///
/// This is the baseline. SIMD and tiled versions will follow.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Matrix-vector multiply: y = A × x
/// A: [M, K], x: [K], y: [M]
pub fn matvec_f32(a: &[f32], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(x.len(), k);
    assert_eq!(y.len(), m);

    for i in 0..m {
        let mut sum = 0.0f32;
        let row = &a[i * k..(i + 1) * k];
        for j in 0..k {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

// TODO: tiled matmul for cache efficiency
// TODO: SIMD (AVX2/NEON) vectorized inner loop
// TODO: quantized matmul (Q4_K × f32)
// TODO: rayon-parallelized row batches

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        // 2x2 identity × [1,2; 3,4] = [1,2; 3,4]
        let a = [1.0, 0.0, 0.0, 1.0f32];
        let b = [1.0, 2.0, 3.0, 4.0f32];
        let mut c = [0.0f32; 4];
        matmul_f32(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matvec() {
        let a = [1.0, 2.0, 3.0, 4.0f32]; // 2x2
        let x = [1.0, 1.0f32];
        let mut y = [0.0f32; 2];
        matvec_f32(&a, &x, &mut y, 2, 2);
        assert_eq!(y, [3.0, 7.0]);
    }
}
