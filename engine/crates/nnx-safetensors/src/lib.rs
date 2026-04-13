//! SafeTensors weight loading for NNX.
//!
//! SafeTensors is HuggingFace's standard format for distributing model weights.
//! It's weights-only (no computation graph) — you need a config.json and
//! architecture code to use the weights.

pub mod loader;

pub use loader::SafeTensorsFile;
