//! Matrix multiplication kernels with SIMD and multithreading.
//!
//! The hot path for transformer inference is matvec (matrix × vector),
//! executed once per weight matrix per token. For a 7B model that's
//! ~20 matvec calls per layer × 32 layers = ~640 matvecs per token.

use rayon::prelude::*;

// ============================================================
// SIMD dot product — the innermost loop of everything
// ============================================================

/// Dot product of two f32 slices. Dispatches to SIMD when available.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 8 {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if a.len() >= 4 {
            return unsafe { dot_f32_neon(a, b) };
        }
    }

    dot_f32_scalar(a, b)
}

/// Scalar fallback dot product.
#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// AVX2 dot product — processes 8 floats per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a_ptr.add(offset));
            let vb = _mm256_loadu_ps(b_ptr.add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        // Horizontal sum of 8 floats in acc
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let mut sum = _mm_cvtss_f32(result);

        // Handle remainder
        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += a[tail_start + i] * b[tail_start + i];
        }

        sum
    }
}

/// NEON dot product for ARM (Apple Silicon, etc.) — 4 floats per iteration.
#[cfg(target_arch = "aarch64")]
unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut sum = vaddvq_f32(acc);

    let tail_start = chunks * 4;
    for i in 0..remainder {
        sum += a[tail_start + i] * b[tail_start + i];
    }

    sum
}

// ============================================================
// Matrix-vector multiply (the workhorse)
// ============================================================

/// Minimum row count to use rayon parallelism (avoid overhead for small ops).
const PARALLEL_ROW_THRESHOLD: usize = 64;

/// Matrix-vector multiply: y = A × x
/// A: [M, K] row-major, x: [K], y: [M]
///
/// Parallelized across rows, SIMD within each dot product.
pub fn matvec_f32(a: &[f32], x: &[f32], y: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), m);

    if m >= PARALLEL_ROW_THRESHOLD {
        // Parallel: each thread computes a chunk of output rows
        y.par_chunks_mut(1).enumerate().for_each(|(i, yi)| {
            let row = &a[i * k..(i + 1) * k];
            yi[0] = dot_f32(row, x);
        });
    } else {
        // Sequential for small matrices
        for i in 0..m {
            let row = &a[i * k..(i + 1) * k];
            y[i] = dot_f32(row, x);
        }
    }
}

/// Matrix-vector multiply with additive bias: y = A × x + bias
pub fn matvec_bias_f32(a: &[f32], x: &[f32], bias: &[f32], y: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(bias.len(), m);
    matvec_f32(a, x, y, m, k);
    for i in 0..m {
        y[i] += bias[i];
    }
}

// ============================================================
// Matrix multiply (for batched operations / prefill)
// ============================================================

/// Matrix multiply: C = A × B
/// A: [M, K], B: [K, N], C: [M, N]
///
/// Parallelized across M rows.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    if m >= PARALLEL_ROW_THRESHOLD {
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            let a_row = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_row[p] * b[p * n + j];
                }
                c_row[j] = sum;
            }
        });
    } else {
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
}

// ============================================================
// Quantized matvec — dequantize on the fly
// ============================================================

/// Quantized matrix-vector multiply: y = dequant(A_quant) × x
///
/// Each row of A is stored in a quantized format. We dequantize one row
/// at a time into a scratch buffer, then dot with x. This keeps memory
/// at ~4x less than f32 storage.
///
/// `row_data`: closure that returns the raw quantized bytes for row `i`
/// `dtype`: quantization format
pub fn matvec_quantized<'a>(
    row_data: impl Fn(usize) -> &'a [u8] + Sync,
    x: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    dtype: nnx_quant::GGMLType,
) {
    if m >= PARALLEL_ROW_THRESHOLD {
        y.par_chunks_mut(1).enumerate().for_each(|(i, yi)| {
            let mut row_buf = vec![0.0f32; k];
            nnx_quant::dequant::dequantize(row_data(i), dtype, &mut row_buf);
            yi[0] = dot_f32(&row_buf, x);
        });
    } else {
        let mut row_buf = vec![0.0f32; k];
        for i in 0..m {
            nnx_quant::dequant::dequantize(row_data(i), dtype, &mut row_buf);
            y[i] = dot_f32(&row_buf, x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let b = vec![1.0; 8];
        let result = dot_f32(&a, &b);
        assert!((result - 36.0).abs() < 1e-4);
    }

    #[test]
    fn test_dot_product_large() {
        // Large enough to trigger SIMD paths
        let n = 4096;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.001).collect();
        let simd_result = dot_f32(&a, &b);
        let scalar_result = dot_f32_scalar(&a, &b);
        assert!(
            (simd_result - scalar_result).abs() < 1.0,
            "SIMD ({}) vs scalar ({}) mismatch",
            simd_result, scalar_result
        );
    }

    #[test]
    fn test_matmul_identity() {
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

    #[test]
    fn test_matvec_large_parallel() {
        // Large enough to trigger parallel path
        let m = 256;
        let k = 128;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|i| (i % 5) as f32 * 0.1).collect();
        let mut y = vec![0.0f32; m];
        matvec_f32(&a, &x, &mut y, m, k);

        // Verify against scalar
        for i in 0..m {
            let expected = dot_f32_scalar(&a[i * k..(i + 1) * k], &x);
            assert!(
                (y[i] - expected).abs() < 1e-3,
                "row {}: got {}, expected {}",
                i, y[i], expected
            );
        }
    }

    #[test]
    fn test_matvec_bias() {
        let a = [1.0, 0.0, 0.0, 1.0f32]; // identity
        let x = [3.0, 5.0f32];
        let bias = [10.0, 20.0f32];
        let mut y = [0.0f32; 2];
        matvec_bias_f32(&a, &x, &bias, &mut y, 2, 2);
        assert_eq!(y, [13.0, 25.0]);
    }
}
