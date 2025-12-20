//! Custom hardware provider framework for NPU, TPU, and other accelerators.
//!
//! This module provides an extensible framework for integrating custom hardware
//! accelerators, including Neural Processing Units (NPUs), Tensor Processing Units (TPUs),
//! and other specialized AI accelerators through a plugin-based architecture.
//!
//! ## Key Features
//! - **Plugin-based architecture**: Dynamic loading of custom providers
//! - **Hardware abstraction**: Unified interface for diverse accelerators
//! - **Capability discovery**: Automatic detection of hardware capabilities
//! - **Custom kernel compilation**: Support for hardware-specific optimizations
//! - **Memory management**: Abstracted memory allocation for custom devices
//! - **Performance profiling**: Built-in profiling and monitoring hooks

#[cfg(feature = "custom-hardware")]
pub mod example_npu;
#[cfg(feature = "custom-hardware")]
pub mod example_tpu;
#[cfg(feature = "custom-hardware")]
pub mod registry;
#[cfg(feature = "custom-hardware")]
pub mod traits;

#[cfg(feature = "custom-hardware")]
pub use example_npu::{NpuConfig, NpuProvider, create_npu_provider};
#[cfg(feature = "custom-hardware")]
pub use example_tpu::{TpuConfig, TpuProvider, create_tpu_provider};
#[cfg(feature = "custom-hardware")]
pub use registry::{CustomProviderRegistry, PluginMetadata, ProviderPlugin};
#[cfg(feature = "custom-hardware")]
pub use traits::{CustomHardwareProvider, CustomKernel, DeviceMemory, HardwareCapability};

#[cfg(not(feature = "custom-hardware"))]
/// Custom hardware provider framework is not available - enable the "custom-hardware" feature.
pub fn custom_hardware_not_available() -> anyhow::Result<()> {
    anyhow::bail!(
        "Custom hardware provider framework not available - enable the 'custom-hardware' feature flag"
    )
}
