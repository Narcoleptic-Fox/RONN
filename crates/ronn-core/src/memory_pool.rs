//! Memory pooling for efficient tensor allocation.
//!
//! Reduces allocation overhead by reusing memory buffers across operations.
//! Critical for low-latency, high-throughput inference.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A memory buffer that can be reused.
#[derive(Clone)]
pub struct PooledBuffer {
    data: Vec<u8>,
    capacity: usize,
}

impl PooledBuffer {
    /// Create a new pooled buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Get the capacity of this buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get a mutable reference to the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get the underlying data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Resize the buffer to the given size.
    pub fn resize(&mut self, new_size: usize, value: u8) {
        self.data.resize(new_size, value);
    }
}

/// Memory pool for efficient allocation and reuse of buffers.
///
/// Maintains separate pools for different size classes to minimize fragmentation.
/// Thread-safe via internal locking.
///
/// # Performance
///
/// - Cache hit: O(1) - reuse existing buffer
/// - Cache miss: O(1) - allocate new buffer
/// - Return: O(1) - add to pool
///
/// # Example
///
/// ```
/// use ronn_core::memory_pool::MemoryPool;
///
/// let pool = MemoryPool::new();
///
/// // Get buffer (cache miss - allocates)
/// let mut buf = pool.get(1024);
/// // Use buffer...
///
/// // Return buffer (adds to pool)
/// pool.return_buffer(buf);
///
/// // Get buffer again (cache hit - reuses)
/// let buf2 = pool.get(1024);
/// ```
pub struct MemoryPool {
    /// Pools organized by size class
    pools: Arc<Mutex<HashMap<usize, Vec<PooledBuffer>>>>,
    /// Statistics
    stats: Arc<Mutex<PoolStats>>,
    /// Configuration
    config: PoolConfig,
}

/// Configuration for memory pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of buffers per size class
    pub max_buffers_per_size: usize,
    /// Maximum total buffers across all sizes
    pub max_total_buffers: usize,
    /// Enable size class rounding (powers of 2)
    pub round_sizes: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_buffers_per_size: 16,
            max_total_buffers: 256,
            round_sizes: true,
        }
    }
}

/// Statistics about memory pool usage.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of get() calls
    pub total_gets: u64,
    /// Number of cache hits (reused buffer)
    pub cache_hits: u64,
    /// Number of cache misses (new allocation)
    pub cache_misses: u64,
    /// Total number of return_buffer() calls
    pub total_returns: u64,
    /// Number of buffers currently in pool
    pub buffers_in_pool: usize,
    /// Total bytes currently pooled
    pub bytes_in_pool: usize,
}

impl PoolStats {
    /// Calculate cache hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        if self.total_gets == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_gets as f64
        }
    }

    /// Calculate cache miss rate (0.0 to 1.0).
    pub fn miss_rate(&self) -> f64 {
        1.0 - self.hit_rate()
    }
}

impl MemoryPool {
    /// Create a new memory pool with default configuration.
    pub fn new() -> Self {
        Self::with_config(PoolConfig::default())
    }

    /// Create a new memory pool with custom configuration.
    pub fn with_config(config: PoolConfig) -> Self {
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(PoolStats::default())),
            config,
        }
    }

    /// Get a buffer of at least the specified size.
    ///
    /// Returns a cached buffer if available, otherwise allocates a new one.
    ///
    /// # Arguments
    ///
    /// * `size` - Minimum size in bytes
    ///
    /// # Returns
    ///
    /// A pooled buffer with capacity >= size
    pub fn get(&self, size: usize) -> PooledBuffer {
        let size_class = self.size_class(size);

        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.total_gets += 1;

        // Try to get from pool
        if let Some(pool) = pools.get_mut(&size_class) {
            if let Some(buffer) = pool.pop() {
                stats.cache_hits += 1;
                stats.buffers_in_pool -= 1;
                stats.bytes_in_pool -= buffer.capacity();
                return buffer;
            }
        }

        // Cache miss - allocate new buffer
        stats.cache_misses += 1;
        PooledBuffer::new(size_class)
    }

    /// Return a buffer to the pool for reuse.
    ///
    /// If the pool for this size class is full, the buffer is dropped.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to return
    pub fn return_buffer(&self, buffer: PooledBuffer) {
        let size_class = buffer.capacity();

        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.total_returns += 1;

        // Check if pool is full
        let pool = pools.entry(size_class).or_insert_with(Vec::new);

        if pool.len() < self.config.max_buffers_per_size
            && stats.buffers_in_pool < self.config.max_total_buffers
        {
            stats.buffers_in_pool += 1;
            stats.bytes_in_pool += buffer.capacity();
            pool.push(buffer);
        }
        // Otherwise drop the buffer (will be freed)
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all buffers from the pool.
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        pools.clear();
        stats.buffers_in_pool = 0;
        stats.bytes_in_pool = 0;
    }

    /// Calculate size class for a given size.
    ///
    /// If round_sizes is enabled, rounds up to nearest power of 2.
    /// Otherwise returns the size as-is.
    fn size_class(&self, size: usize) -> usize {
        if self.config.round_sizes {
            size.next_power_of_two()
        } else {
            size
        }
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Global memory pool instance.
///
/// Thread-safe singleton for sharing across the application.
static GLOBAL_POOL: once_cell::sync::Lazy<MemoryPool> =
    once_cell::sync::Lazy::new(|| MemoryPool::new());

/// Get the global memory pool instance.
pub fn global_pool() -> &'static MemoryPool {
    &GLOBAL_POOL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = MemoryPool::new();
        let stats = pool.stats();
        assert_eq!(stats.total_gets, 0);
        assert_eq!(stats.cache_hits, 0);
    }

    #[test]
    fn test_get_and_return() {
        let pool = MemoryPool::new();

        // First get - cache miss
        let buf = pool.get(1024);
        assert_eq!(buf.capacity(), 1024);

        let stats = pool.stats();
        assert_eq!(stats.total_gets, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 0);

        // Return buffer
        pool.return_buffer(buf);

        let stats = pool.stats();
        assert_eq!(stats.total_returns, 1);
        assert_eq!(stats.buffers_in_pool, 1);

        // Second get - cache hit
        let buf2 = pool.get(1024);
        assert_eq!(buf2.capacity(), 1024);

        let stats = pool.stats();
        assert_eq!(stats.total_gets, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_size_rounding() {
        let pool = MemoryPool::new();

        // Request 1000 bytes, should round to 1024
        let buf = pool.get(1000);
        assert_eq!(buf.capacity(), 1024);
    }

    #[test]
    fn test_pool_limit() {
        let config = PoolConfig {
            max_buffers_per_size: 2,
            ..Default::default()
        };
        let pool = MemoryPool::with_config(config);

        // Add 3 buffers, only 2 should be kept
        pool.return_buffer(PooledBuffer::new(1024));
        pool.return_buffer(PooledBuffer::new(1024));
        pool.return_buffer(PooledBuffer::new(1024));

        let stats = pool.stats();
        assert_eq!(stats.buffers_in_pool, 2);
    }

    #[test]
    fn test_multiple_sizes() {
        let pool = MemoryPool::new();

        let buf1 = pool.get(1024);
        let buf2 = pool.get(2048);
        let buf3 = pool.get(4096);

        pool.return_buffer(buf1);
        pool.return_buffer(buf2);
        pool.return_buffer(buf3);

        let stats = pool.stats();
        assert_eq!(stats.buffers_in_pool, 3);
        assert_eq!(stats.bytes_in_pool, 1024 + 2048 + 4096);
    }

    #[test]
    fn test_clear() {
        let pool = MemoryPool::new();

        pool.return_buffer(PooledBuffer::new(1024));
        pool.return_buffer(PooledBuffer::new(2048));

        assert_eq!(pool.stats().buffers_in_pool, 2);

        pool.clear();

        assert_eq!(pool.stats().buffers_in_pool, 0);
        assert_eq!(pool.stats().bytes_in_pool, 0);
    }

    #[test]
    fn test_global_pool() {
        let pool1 = global_pool();
        let pool2 = global_pool();

        // Should be the same instance
        assert!(std::ptr::eq(pool1, pool2));
    }
}
