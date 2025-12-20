//! WebAssembly execution provider for browser and edge deployment.
//!
//! This module provides a WebAssembly-compatible execution provider that can
//! run neural network inference in browsers and edge environments with
//! minimal overhead and optimal memory usage.
//!
//! ## Key Features
//! - **Browser compatibility**: Runs in all modern browsers
//! - **WASM SIMD128**: Uses WebAssembly SIMD when available
//! - **Memory efficiency**: Optimized for WASM linear memory constraints
//! - **JavaScript interop**: Seamless integration with TypedArrays
//! - **WebGL/WebGPU fallback**: GPU acceleration when available
//! - **Progressive loading**: Streaming model loading with IndexedDB caching

#[cfg(feature = "wasm")]
pub mod allocator;
#[cfg(feature = "wasm")]
pub mod bridge;
#[cfg(feature = "wasm")]
pub mod kernels;
#[cfg(feature = "wasm")]
pub mod provider;

#[cfg(feature = "wasm")]
pub use allocator::WasmMemoryAllocator;
#[cfg(feature = "wasm")]
pub use bridge::{IndexedDbCache, TypedArrayInterface, WasmBridge};
#[cfg(feature = "wasm")]
pub use kernels::{WasmKernel, WasmSimd128Ops};
#[cfg(feature = "wasm")]
pub use provider::{WasmExecutionProvider, WasmProviderConfig, create_wasm_provider};

#[cfg(not(feature = "wasm"))]
/// WebAssembly provider is not available - enable the "wasm" feature to use it.
pub fn wasm_not_available() -> anyhow::Result<()> {
    anyhow::bail!("WebAssembly provider not available - enable the 'wasm' feature flag")
}
