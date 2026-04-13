//! Optimized compute kernels for NNX.
//!
//! Pure Rust implementations of the core operations needed for transformer
//! inference. SIMD-accelerated where possible.

pub mod activations;
pub mod matmul;
pub mod rope;
pub mod rms_norm;
pub mod softmax;
