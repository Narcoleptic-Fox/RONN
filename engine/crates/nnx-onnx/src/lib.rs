//! ONNX model loading for NNX.
//!
//! Provides ONNX protobuf parsing and graph construction. This crate handles
//! format loading only — the existing `ronn-onnx` crate in the RONN workspace
//! can be refactored to use this as its foundation.

pub mod loader;

// TODO: Implement ONNX protobuf parser.
// For now, this is a placeholder — the existing ronn-onnx crate already has
// a working parser that can be migrated here.
