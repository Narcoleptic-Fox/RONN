//! Serving configuration.

/// GPU KV-cache quantization settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheQuantizationConfig {
    /// Enable paged GPU KV quantization.
    pub enabled: bool,
    /// Residual sketch width used for key-score correction.
    pub residual_sketch_dim: usize,
}

impl Default for KvCacheQuantizationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            residual_sketch_dim: 16,
        }
    }
}

/// Configuration for the serving infrastructure.
#[derive(Debug, Clone)]
pub struct ServingConfig {
    /// Number of tokens per KV cache page. Must be a power of 2.
    /// Default: 16 (vLLM standard, good cache-line alignment).
    pub page_size: usize,

    /// Maximum number of physical pages in the block allocator.
    /// Determines total KV cache memory budget.
    /// 0 = auto-calculate from available memory.
    pub max_pages: usize,

    /// Maximum number of sequences that can be active simultaneously.
    pub max_sequences: usize,

    /// Maximum number of sequences in a single forward batch.
    pub max_batch_size: usize,

    /// Whether to enable prefix caching.
    pub enable_prefix_caching: bool,

    /// Maximum number of page-groups in the prefix cache.
    /// When the cache exceeds this limit, LRU entries are evicted and their
    /// pages freed. 0 = no limit (NOT recommended for production — will
    /// eventually exhaust the page allocator under diverse-prompt traffic).
    pub max_prefix_cache_entries: usize,

    /// Maximum number of tokens to prefill in a single iteration (chunked prefill).
    /// 0 = unlimited (prefill entire prompt at once).
    pub max_prefill_tokens: usize,

    /// Optional GPU KV-cache quantization.
    pub gpu_kv_quantization: KvCacheQuantizationConfig,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            page_size: 16,
            max_pages: 0,
            max_sequences: 256,
            max_batch_size: 32,
            enable_prefix_caching: true,
            max_prefix_cache_entries: 1024,
            max_prefill_tokens: 0,
            gpu_kv_quantization: KvCacheQuantizationConfig::default(),
        }
    }
}

impl ServingConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.page_size == 0 {
            return Err("page_size must be > 0".into());
        }
        if !self.page_size.is_power_of_two() {
            return Err(format!(
                "page_size must be a power of 2, got {}",
                self.page_size
            ));
        }
        if self.max_sequences == 0 {
            return Err("max_sequences must be > 0".into());
        }
        if self.max_batch_size == 0 {
            return Err("max_batch_size must be > 0".into());
        }
        if self.gpu_kv_quantization.enabled && self.gpu_kv_quantization.residual_sketch_dim != 16 {
            return Err(
                "gpu_kv_quantization.residual_sketch_dim must be 16 for the current GPU KV quantization path"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        ServingConfig::default().validate().unwrap();
    }

    #[test]
    fn rejects_zero_page_size() {
        let mut cfg = ServingConfig::default();
        cfg.page_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_non_power_of_two_page_size() {
        let mut cfg = ServingConfig::default();
        cfg.page_size = 15;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn accepts_valid_page_sizes() {
        for &size in &[1, 2, 4, 8, 16, 32, 64] {
            let mut cfg = ServingConfig::default();
            cfg.page_size = size;
            cfg.validate().unwrap();
        }
    }

    #[test]
    fn rejects_zero_residual_sketch_dim_when_quantization_enabled() {
        let mut cfg = ServingConfig::default();
        cfg.gpu_kv_quantization.enabled = true;
        cfg.gpu_kv_quantization.residual_sketch_dim = 0;
        assert!(cfg.validate().is_err());
    }
}
