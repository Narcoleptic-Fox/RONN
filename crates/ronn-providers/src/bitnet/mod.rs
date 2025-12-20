//! BitNet execution provider module for 1-bit quantized model support.
//!
//! BitNet is a revolutionary approach to neural network quantization that reduces
//! model weights to 1 bit while maintaining competitive accuracy. This provider
//! implements highly optimized 1-bit operations using bit manipulation and SIMD.
//!
//! ## Key Features
//! - **1-bit quantization**: Weights compressed to {-1, +1} values
//! - **1.58-bit quantization**: Extended support for {-1, 0, +1} ternary values
//! - **Bit-packed storage**: 8x memory reduction compared to FP32
//! - **SIMD-optimized kernels**: XNOR and popcount operations for ultra-fast inference
//! - **Mixed precision**: 1-bit weights with higher precision activations

#[cfg(feature = "bitnet")]
pub mod allocator;
#[cfg(feature = "bitnet")]
pub mod kernels;
#[cfg(feature = "bitnet")]
pub mod provider;
#[cfg(feature = "bitnet")]
pub mod quantization;

#[cfg(feature = "bitnet")]
pub use allocator::BitNetMemoryAllocator;
#[cfg(feature = "bitnet")]
pub use kernels::{BitNetKernel, BitNetOperation};
#[cfg(feature = "bitnet")]
pub use provider::{BitNetExecutionProvider, BitNetProviderConfig, create_bitnet_provider};
#[cfg(feature = "bitnet")]
pub use quantization::{BinaryTensor, BitNetQuantizer, QuantizationMethod, TernaryTensor};

#[cfg(not(feature = "bitnet"))]
/// BitNet provider is not available - enable the "bitnet" feature to use it.
pub fn bitnet_not_available() -> anyhow::Result<()> {
    anyhow::bail!("BitNet provider not available - enable the 'bitnet' feature flag")
}
