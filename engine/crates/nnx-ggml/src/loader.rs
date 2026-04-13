//! GGML binary file loader.
//!
//! GGML files have a simpler structure than GGUF:
//! - Magic number (0x67676d6c for "ggml" or versioned variants)
//! - Hyperparameters (architecture-specific, no standard layout)
//! - Vocabulary
//! - Tensor data
//!
//! Because GGML has no standardized metadata, the caller must provide
//! architecture information (number of layers, hidden dim, etc.).

use nnx_core::error::{EngineError, Result};
use std::path::Path;
use tracing::warn;

/// Known GGML magic numbers.
pub const GGML_MAGIC: u32 = 0x67676D6C; // "ggml"
pub const GGML_MAGIC_V1: u32 = 0x67676D66; // "ggmf" (versioned)
pub const GGML_MAGIC_V2: u32 = 0x67676A74; // "ggjt" (version 2)

/// Architecture hint — since GGML files don't self-describe, the caller
/// must tell us what architecture to expect.
#[derive(Debug, Clone)]
pub struct GGMLArchHint {
    pub architecture: String,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
}

/// A parsed GGML file.
pub struct GGMLFile {
    pub magic: u32,
    pub arch: GGMLArchHint,
    mmap: memmap2::Mmap,
}

impl GGMLFile {
    /// Open a GGML file with an architecture hint.
    pub fn open(path: &Path, arch: GGMLArchHint) -> Result<Self> {
        warn!(
            "Loading legacy GGML format from {}. Consider converting to GGUF.",
            path.display()
        );

        let file = std::fs::File::open(path).map_err(|e| {
            EngineError::ModelLoad(format!("failed to open {}: {}", path.display(), e))
        })?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                EngineError::ModelLoad(format!("failed to mmap {}: {}", path.display(), e))
            })?
        };

        let data = &mmap[..];
        if data.len() < 4 {
            return Err(EngineError::ModelLoad("file too small".into()));
        }

        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != GGML_MAGIC && magic != GGML_MAGIC_V1 && magic != GGML_MAGIC_V2 {
            return Err(EngineError::UnsupportedFormat(format!(
                "not a GGML file (magic: 0x{:08X})",
                magic
            )));
        }

        Ok(Self { magic, arch, mmap })
    }

    /// File size in bytes.
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    // TODO: implement tensor extraction based on architecture hint.
    // GGML layout is architecture-dependent, so each architecture
    // (llama, gpt2, etc.) needs its own parsing logic.
}
