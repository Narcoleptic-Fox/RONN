//! GGUF metadata access helpers.
//!
//! Provides typed access to well-known GGUF metadata keys for model
//! architecture, tokenizer, and quantization information.

use crate::types::GGUFValue;
use std::collections::HashMap;

/// Parsed GGUF metadata with convenience accessors.
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    /// Raw key-value pairs.
    pub values: HashMap<String, GGUFValue>,
}

impl GGUFMetadata {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Get a raw value by key.
    pub fn get(&self, key: &str) -> Option<&GGUFValue> {
        self.values.get(key)
    }

    // -- Architecture keys --

    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture")?.as_str()
    }

    pub fn name(&self) -> Option<&str> {
        self.get("general.name")?.as_str()
    }

    pub fn num_layers(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.block_count"))?.as_u32()
    }

    pub fn hidden_dim(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.embedding_length"))?.as_u32()
    }

    pub fn num_heads(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.attention.head_count"))?.as_u32()
    }

    pub fn num_kv_heads(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.attention.head_count_kv"))?.as_u32()
    }

    pub fn context_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.context_length"))?.as_u32()
    }

    pub fn vocab_size(&self) -> Option<u32> {
        // Try tokenizer key first, then architecture key
        self.get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GGUFValue::Array(arr) => Some(arr.len() as u32),
                _ => None,
            })
    }

    pub fn feed_forward_dim(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.feed_forward_length"))?.as_u32()
    }

    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        self.get(&format!("{arch}.rope.freq_base"))?.as_f32()
    }
}

impl Default for GGUFMetadata {
    fn default() -> Self {
        Self::new()
    }
}
