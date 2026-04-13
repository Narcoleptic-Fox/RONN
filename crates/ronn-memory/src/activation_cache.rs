//! Activation pattern cache for repeated prompt prefixes.
//!
//! Caches predictor outputs (activation masks) keyed by a hash of the
//! hidden state input. When the same prompt prefix is seen again, the
//! cached activation patterns can be reused without re-running the predictor,
//! saving the prediction overhead.

use crate::{MemoryError, Result};
use std::collections::HashMap;

/// A cached activation pattern for one FFN layer.
#[derive(Debug, Clone)]
pub struct CachedActivationPattern {
    /// Layer this pattern belongs to.
    pub layer_id: usize,
    /// Active neuron indices predicted for this input.
    pub active_indices: Vec<usize>,
    /// Timestamp when this pattern was cached (ms since epoch).
    pub cached_at: u64,
    /// Number of times this cached pattern has been reused.
    pub hit_count: u64,
}

/// Key for the activation cache: a hash of the hidden state tensor.
///
/// We use a simple hash of the first few values and shape to avoid
/// hashing the entire tensor (which could be large).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ActivationCacheKey {
    /// Layer ID.
    pub layer_id: usize,
    /// Hash of the hidden state.
    pub input_hash: u64,
}

impl ActivationCacheKey {
    /// Create a cache key from a layer ID and hidden state data.
    pub fn new(layer_id: usize, hidden_state_data: &[f32]) -> Self {
        let input_hash = Self::hash_f32_slice(hidden_state_data);
        Self {
            layer_id,
            input_hash,
        }
    }

    /// Simple hash of f32 slice (not cryptographic, just for cache lookup).
    fn hash_f32_slice(data: &[f32]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        let prime: u64 = 0x100000001b3;

        // Hash the length
        hash ^= data.len() as u64;
        hash = hash.wrapping_mul(prime);

        // Hash samples: first 16, last 16, and every Nth element
        let stride = (data.len() / 32).max(1);
        for i in (0..data.len()).step_by(stride).take(64) {
            let bits = data[i].to_bits() as u64;
            hash ^= bits;
            hash = hash.wrapping_mul(prime);
        }

        hash
    }
}

/// Configuration for the activation cache.
#[derive(Debug, Clone)]
pub struct ActivationCacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Time-to-live for cache entries in milliseconds.
    pub ttl_ms: u64,
}

impl Default for ActivationCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            ttl_ms: 300_000, // 5 minutes
        }
    }
}

/// Cache for activation patterns that can be reused across inference calls.
pub struct ActivationCache {
    config: ActivationCacheConfig,
    entries: HashMap<ActivationCacheKey, CachedActivationPattern>,
    /// Total cache lookups.
    total_lookups: u64,
    /// Cache hits.
    cache_hits: u64,
}

impl ActivationCache {
    /// Create a new activation cache.
    pub fn new(config: ActivationCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            total_lookups: 0,
            cache_hits: 0,
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(ActivationCacheConfig::default())
    }

    /// Look up a cached activation pattern.
    pub fn get(
        &mut self,
        layer_id: usize,
        hidden_state_data: &[f32],
    ) -> Option<&CachedActivationPattern> {
        self.total_lookups += 1;
        let key = ActivationCacheKey::new(layer_id, hidden_state_data);

        if let Some(entry) = self.entries.get_mut(&key) {
            let now = crate::current_timestamp();
            if now - entry.cached_at < self.config.ttl_ms {
                entry.hit_count += 1;
                self.cache_hits += 1;
                // Re-borrow as immutable
                return self.entries.get(&key);
            } else {
                // Expired, remove
                self.entries.remove(&key);
                return None;
            }
        }

        None
    }

    /// Store an activation pattern in the cache.
    pub fn put(
        &mut self,
        layer_id: usize,
        hidden_state_data: &[f32],
        active_indices: Vec<usize>,
    ) {
        // Evict if at capacity
        if self.entries.len() >= self.config.max_entries {
            self.evict_lru();
        }

        let key = ActivationCacheKey::new(layer_id, hidden_state_data);
        let pattern = CachedActivationPattern {
            layer_id,
            active_indices,
            cached_at: crate::current_timestamp(),
            hit_count: 0,
        };
        self.entries.insert(key, pattern);
    }

    /// Evict the least-recently-used (lowest hit count) entry.
    fn evict_lru(&mut self) {
        if let Some(key_to_remove) = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.hit_count)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&key_to_remove);
        }
    }

    /// Get cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.total_lookups as f64
    }

    /// Get the number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_lookups = 0;
        self.cache_hits = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> ActivationCacheStats {
        ActivationCacheStats {
            entries: self.entries.len(),
            max_entries: self.config.max_entries,
            total_lookups: self.total_lookups,
            cache_hits: self.cache_hits,
            hit_rate: self.hit_rate(),
        }
    }
}

/// Statistics about the activation cache.
#[derive(Debug, Clone)]
pub struct ActivationCacheStats {
    pub entries: usize,
    pub max_entries: usize,
    pub total_lookups: u64,
    pub cache_hits: u64,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_hashing() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![1.0, 2.0, 3.0, 5.0]; // different
        let data3 = vec![1.0, 2.0, 3.0, 4.0]; // same as data1

        let key1 = ActivationCacheKey::new(0, &data1);
        let key2 = ActivationCacheKey::new(0, &data2);
        let key3 = ActivationCacheKey::new(0, &data3);

        assert_eq!(key1, key3); // Same data = same key
        assert_ne!(key1, key2); // Different data = different key

        // Different layer = different key
        let key4 = ActivationCacheKey::new(1, &data1);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_cache_put_get() {
        let mut cache = ActivationCache::with_defaults();

        let data = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 2, 5];

        cache.put(0, &data, indices.clone());
        assert_eq!(cache.len(), 1);

        let result = cache.get(0, &data);
        assert!(result.is_some());
        assert_eq!(result.unwrap().active_indices, indices);
        assert_eq!(result.unwrap().hit_count, 1);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = ActivationCache::with_defaults();

        let data1 = vec![1.0, 2.0, 3.0];
        let data2 = vec![4.0, 5.0, 6.0];

        cache.put(0, &data1, vec![0, 1]);

        let result = cache.get(0, &data2);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let config = ActivationCacheConfig {
            max_entries: 2,
            ttl_ms: 300_000,
        };
        let mut cache = ActivationCache::new(config);

        cache.put(0, &[1.0], vec![0]);
        cache.put(0, &[2.0], vec![1]);
        assert_eq!(cache.len(), 2);

        // Adding a third should evict one
        cache.put(0, &[3.0], vec![2]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = ActivationCache::with_defaults();
        let data = vec![1.0, 2.0];

        cache.put(0, &data, vec![0]);
        cache.get(0, &data); // hit
        cache.get(0, &[9.0]); // miss

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.cache_hits, 1);
        assert!((stats.hit_rate - 0.5).abs() < f64::EPSILON);
    }
}
