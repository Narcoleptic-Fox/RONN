//! Legacy GGML format loading (optional compatibility).
//!
//! GGML (`.bin`) is the predecessor to GGUF. It lacks self-describing metadata,
//! so the loader requires the model architecture to be specified externally.
//! Provided for backwards compatibility with older model files.
//!
//! For new models, use GGUF instead.

pub mod loader;

pub use loader::{GGMLArchHint, GGMLFile, GGMLHeader, GGMLTensorInfo, GGMLTensorView};
