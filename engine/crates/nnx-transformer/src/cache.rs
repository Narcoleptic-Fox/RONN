//! KV cache for autoregressive generation.

use nnx_core::error::{EngineError, Result};

/// Per-layer KV cache storing keys and values as contiguous f32 buffers.
#[derive(Debug, Clone)]
pub struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    len: usize,
}

impl LayerCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let size = max_seq_len * num_kv_heads * head_dim;
        Self {
            keys: vec![0.0; size],
            values: vec![0.0; size],
            num_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Store key/value at current position. Both have shape [num_kv_heads * head_dim].
    pub fn store(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        let kv_size = self.num_kv_heads * self.head_dim;
        if key.len() != kv_size {
            return Err(EngineError::ShapeMismatch(format!(
                "key length {} does not match expected kv size {}",
                key.len(),
                kv_size
            )));
        }
        if value.len() != kv_size {
            return Err(EngineError::ShapeMismatch(format!(
                "value length {} does not match expected kv size {}",
                value.len(),
                kv_size
            )));
        }
        if self.len >= self.max_seq_len {
            return Err(EngineError::Cache(format!(
                "KV cache full: position {} exceeds capacity {}",
                self.len, self.max_seq_len
            )));
        }

        let offset = self.len * kv_size;
        self.keys[offset..offset + kv_size].copy_from_slice(key);
        self.values[offset..offset + kv_size].copy_from_slice(value);
        self.len += 1;
        Ok(())
    }

    pub fn key_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let offset = pos * self.num_kv_heads * self.head_dim + kv_head * self.head_dim;
        &self.keys[offset..offset + self.head_dim]
    }

    pub fn value_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let offset = pos * self.num_kv_heads * self.head_dim + kv_head * self.head_dim;
        &self.values[offset..offset + self.head_dim]
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
    pub fn truncate(&mut self, n: usize) {
        self.len = n.min(self.len);
    }

    pub fn memory_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * 4
    }
}

/// Full model KV cache — one LayerCache per transformer layer.
#[derive(Debug, Clone)]
pub struct KVCache {
    layers: Vec<LayerCache>,
}

impl KVCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerCache::new(max_seq_len, num_kv_heads, head_dim))
                .collect(),
        }
    }

    pub fn layer(&self, i: usize) -> &LayerCache {
        &self.layers[i]
    }
    pub fn layer_mut(&mut self, i: usize) -> &mut LayerCache {
        &mut self.layers[i]
    }
    pub fn position(&self) -> usize {
        self.layers.first().map_or(0, |l| l.len())
    }

    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
    }
    pub fn truncate(&mut self, n: usize) {
        for l in &mut self.layers {
            l.truncate(n);
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Maximum capacity in tokens.
    pub fn capacity(&self) -> usize {
        self.layers.first().map_or(0, |l| l.max_seq_len)
    }
}

impl nnx_core::engine::KVCacheAccess for KVCache {
    fn cached_tokens(&self) -> usize {
        self.position()
    }
    fn capacity(&self) -> usize {
        KVCache::capacity(self)
    }
    fn memory_usage_bytes(&self) -> usize {
        self.memory_bytes()
    }
    fn clear(&mut self) {
        KVCache::clear(self)
    }
    fn truncate(&mut self, n: usize) {
        KVCache::truncate(self, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_cache() {
        let mut cache = LayerCache::new(16, 2, 4);
        let key = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let val = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        cache.store(&key, &val).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.key_at(0, 0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.key_at(0, 1), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_layer_cache_rejects_overflow() {
        let mut cache = LayerCache::new(1, 1, 2);
        cache.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        let err = cache.store(&[5.0, 6.0], &[7.0, 8.0]).unwrap_err();
        assert!(matches!(err, EngineError::Cache(_)));
    }
}
