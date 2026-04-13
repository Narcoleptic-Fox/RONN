//! Device abstraction for compute targets.

use serde::{Deserialize, Serialize};

/// Identifier for a specific device instance.
pub type DeviceId = u32;

/// Compute device where tensors live and operations execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    /// CPU execution.
    Cpu,
    /// GPU execution (device index for multi-GPU).
    Gpu(DeviceId),
    /// Unified memory (e.g., Apple Silicon — single address space).
    Unified(DeviceId),
}

impl Device {
    /// Whether this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Whether this is a GPU device.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Gpu(_))
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Gpu(id) => write!(f, "gpu:{}", id),
            Device::Unified(id) => write!(f, "unified:{}", id),
        }
    }
}
