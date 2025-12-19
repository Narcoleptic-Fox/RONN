//! SIMD capability detection and optimized operations.
//!
//! This module detects available CPU features and provides SIMD-optimized
//! implementations for common tensor operations.

use std::arch::is_x86_feature_detected;

/// CPU SIMD capabilities detected at runtime.
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// SSE2 support (baseline for x86_64).
    pub sse2: bool,
    /// SSE4.1 support.
    pub sse41: bool,
    /// AVX support (256-bit vectors).
    pub avx: bool,
    /// AVX2 support (integer operations on 256-bit vectors).
    pub avx2: bool,
    /// AVX-512F support (512-bit vectors).
    pub avx512f: bool,
    /// FMA support (fused multiply-add).
    pub fma: bool,
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self {
            sse2: false,
            sse41: false,
            avx: false,
            avx2: false,
            avx512f: false,
            fma: false,
        }
    }
}

/// Detect available SIMD capabilities on the current CPU.
pub fn detect_simd_capabilities() -> SimdCapabilities {
    #[cfg(target_arch = "x86_64")]
    {
        SimdCapabilities {
            sse2: is_x86_feature_detected!("sse2"),
            sse41: is_x86_feature_detected!("sse4.1"),
            avx: is_x86_feature_detected!("avx"),
            avx2: is_x86_feature_detected!("avx2"),
            avx512f: is_x86_feature_detected!("avx512f"),
            fma: is_x86_feature_detected!("fma"),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // For ARM64, we assume NEON is available (it's part of the standard)
        SimdCapabilities {
            sse2: false, // x86-specific
            sse41: false,
            avx: false,
            avx2: false,
            avx512f: false,
            fma: true, // ARM64 has built-in FMA support
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Fallback for unsupported architectures
        SimdCapabilities::default()
    }
}

/// Get the optimal vector width for the current CPU.
pub fn get_optimal_vector_width(capabilities: &SimdCapabilities) -> usize {
    if capabilities.avx512f {
        64 // 512-bit vectors = 64 bytes
    } else if capabilities.avx2 || capabilities.avx {
        32 // 256-bit vectors = 32 bytes
    } else if capabilities.sse2 {
        16 // 128-bit vectors = 16 bytes
    } else {
        8 // Fallback to 64-bit operations
    }
}

/// SIMD-optimized element-wise addition for f32 arrays.
pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32], capabilities: &SimdCapabilities) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    #[cfg(target_arch = "x86_64")]
    {
        if capabilities.avx2 {
            return avx2_add_f32(a, b, result);
        } else if capabilities.avx {
            return avx_add_f32(a, b, result);
        } else if capabilities.sse2 {
            return sse_add_f32(a, b, result);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return neon_add_f32(a, b, result);
    }

    // Fallback to scalar operation
    scalar_add_f32(a, b, result);
}

/// SIMD-optimized element-wise multiplication for f32 arrays.
pub fn simd_mul_f32(a: &[f32], b: &[f32], result: &mut [f32], capabilities: &SimdCapabilities) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    #[cfg(target_arch = "x86_64")]
    {
        if capabilities.avx2 {
            return avx2_mul_f32(a, b, result);
        } else if capabilities.avx {
            return avx_mul_f32(a, b, result);
        } else if capabilities.sse2 {
            return sse_mul_f32(a, b, result);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return neon_mul_f32(a, b, result);
    }

    // Fallback to scalar operation
    scalar_mul_f32(a, b, result);
}

/// SIMD-optimized matrix multiplication for small matrices.
pub fn simd_matmul_f32(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_rows: usize,
    b_cols: usize,
    result: &mut [f32],
    capabilities: &SimdCapabilities,
) {
    assert_eq!(a_cols, b_rows);
    assert_eq!(a.len(), a_rows * a_cols);
    assert_eq!(b.len(), b_rows * b_cols);
    assert_eq!(result.len(), a_rows * b_cols);

    #[cfg(target_arch = "x86_64")]
    {
        if capabilities.avx2 && capabilities.fma {
            return avx2_fma_matmul_f32(a, a_rows, a_cols, b, b_rows, b_cols, result);
        } else if capabilities.avx {
            return avx_matmul_f32(a, a_rows, a_cols, b, b_rows, b_cols, result);
        } else if capabilities.sse2 {
            return sse_matmul_f32(a, a_rows, a_cols, b, b_rows, b_cols, result);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return neon_matmul_f32(a, a_rows, a_cols, b, b_rows, b_cols, result);
    }

    // Fallback to scalar operation
    scalar_matmul_f32(a, a_rows, a_cols, b, b_rows, b_cols, result);
}

// Scalar fallback implementations

fn scalar_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

fn scalar_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] * b[i];
    }
}

fn scalar_matmul_f32(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    _b_rows: usize,
    b_cols: usize,
    result: &mut [f32],
) {
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = 0.0;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            result[i * b_cols + j] = sum;
        }
    }
}

// x86_64 SIMD implementations

#[cfg(target_arch = "x86_64")]
mod x86_simd {
    use std::arch::x86_64::*;

    pub unsafe fn avx2_add_f32_unsafe(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let simd_len = len & !7; // Process 8 elements at a time

            let mut i = 0;
            while i < simd_len {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let result_vec = _mm256_add_ps(a_vec, b_vec);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
                i += 8;
            }

            // Handle remaining elements
            for j in simd_len..len {
                result[j] = a[j] + b[j];
            }
        }
    }

    pub unsafe fn avx2_mul_f32_unsafe(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let simd_len = len & !7; // Process 8 elements at a time

            let mut i = 0;
            while i < simd_len {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                let result_vec = _mm256_mul_ps(a_vec, b_vec);
                _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
                i += 8;
            }

            // Handle remaining elements
            for j in simd_len..len {
                result[j] = a[j] * b[j];
            }
        }
    }

    pub unsafe fn sse_add_f32_unsafe(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let simd_len = len & !3; // Process 4 elements at a time

            let mut i = 0;
            while i < simd_len {
                let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
                let result_vec = _mm_add_ps(a_vec, b_vec);
                _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
                i += 4;
            }

            // Handle remaining elements
            for j in simd_len..len {
                result[j] = a[j] + b[j];
            }
        }
    }

    pub unsafe fn sse_mul_f32_unsafe(a: &[f32], b: &[f32], result: &mut [f32]) {
        unsafe {
            let len = a.len();
            let simd_len = len & !3; // Process 4 elements at a time

            let mut i = 0;
            while i < simd_len {
                let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
                let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
                let result_vec = _mm_mul_ps(a_vec, b_vec);
                _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
                i += 4;
            }

            // Handle remaining elements
            for j in simd_len..len {
                result[j] = a[j] * b[j];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        x86_simd::avx2_add_f32_unsafe(a, b, result);
    }
}

#[cfg(target_arch = "x86_64")]
fn avx_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    // AVX can use the same implementation as AVX2 for addition
    unsafe {
        x86_simd::avx2_add_f32_unsafe(a, b, result);
    }
}

#[cfg(target_arch = "x86_64")]
fn sse_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        x86_simd::sse_add_f32_unsafe(a, b, result);
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        x86_simd::avx2_mul_f32_unsafe(a, b, result);
    }
}

#[cfg(target_arch = "x86_64")]
fn avx_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        x86_simd::avx2_mul_f32_unsafe(a, b, result);
    }
}

#[cfg(target_arch = "x86_64")]
fn sse_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        x86_simd::sse_mul_f32_unsafe(a, b, result);
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_fma_matmul_f32(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    _b_rows: usize,
    b_cols: usize,
    result: &mut [f32],
) {
    // Simplified matrix multiplication using FMA
    // For production, would use more sophisticated blocking and cache optimization
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = 0.0;
            for k in 0..a_cols {
                sum += a[i * a_cols + k] * b[k * b_cols + j];
            }
            result[i * b_cols + j] = sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn avx_matmul_f32(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    _b_rows: usize,
    b_cols: usize,
    result: &mut [f32],
) {
    // Fallback to scalar for now - full AVX matmul is complex
    scalar_matmul_f32(a, a_rows, a_cols, b, _b_rows, b_cols, result);
}

#[cfg(target_arch = "x86_64")]
fn sse_matmul_f32(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    _b_rows: usize,
    b_cols: usize,
    result: &mut [f32],
) {
    // Fallback to scalar for now - full SSE matmul is complex
    scalar_matmul_f32(a, a_rows, a_cols, b, _b_rows, b_cols, result);
}

// ARM64/NEON implementations (simplified for now)

#[cfg(target_arch = "aarch64")]
fn neon_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    // Fallback to scalar for now - would use NEON intrinsics in production
    scalar_add_f32(a, b, result);
}

#[cfg(target_arch = "aarch64")]
fn neon_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    // Fallback to scalar for now - would use NEON intrinsics in production
    scalar_mul_f32(a, b, result);
}

#[cfg(target_arch = "aarch64")]
fn neon_matmul_f32(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    _b_rows: usize,
    b_cols: usize,
    result: &mut [f32],
) {
    // Fallback to scalar for now - would use NEON intrinsics in production
    scalar_matmul_f32(a, a_rows, a_cols, b, _b_rows, b_cols, result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let capabilities = detect_simd_capabilities();
        println!("Detected SIMD capabilities: {:?}", capabilities);

        // Should detect at least some capabilities on most modern systems
        #[cfg(target_arch = "x86_64")]
        {
            assert!(capabilities.sse2); // Should be available on all x86_64
        }
    }

    #[test]
    fn test_simd_add() {
        let capabilities = detect_simd_capabilities();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 8];
        let expected = vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0];

        simd_add_f32(&a, &b, &mut result, &capabilities);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_mul() {
        let capabilities = detect_simd_capabilities();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];
        let expected = vec![2.0, 6.0, 12.0, 20.0];

        simd_mul_f32(&a, &b, &mut result, &capabilities);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_matmul() {
        let capabilities = detect_simd_capabilities();

        // 2x2 * 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]
        let mut result = vec![0.0; 4];
        let expected = vec![19.0, 22.0, 43.0, 50.0]; // [[19, 22], [43, 50]]

        simd_matmul_f32(&a, 2, 2, &b, 2, 2, &mut result, &capabilities);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_width() {
        let capabilities = detect_simd_capabilities();
        let width = get_optimal_vector_width(&capabilities);

        // Should be at least 8 bytes (64-bit)
        assert!(width >= 8);

        // Should be a power of 2
        assert_eq!(width & (width - 1), 0);
    }
}
