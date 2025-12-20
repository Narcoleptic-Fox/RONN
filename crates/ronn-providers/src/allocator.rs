//! Memory allocator implementations for execution providers.
//!
//! This module provides various memory allocation strategies including
//! system memory, pooled memory, and SIMD-aligned allocations.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Result, anyhow};
use ronn_core::{DataType, MemoryInfo, MemoryType, TensorAllocator, TensorBuffer};
use tracing::debug;

/// Calculate the size in bytes for a tensor with given shape and data type.
pub fn calculate_tensor_size(shape: &[usize], dtype: DataType) -> usize {
    let element_count: usize = shape.iter().product();
    let element_size = match dtype {
        DataType::F32 | DataType::I32 | DataType::U32 => 4,
        DataType::F16 | DataType::BF16 => 2,
        DataType::F64 | DataType::I64 => 8,
        DataType::I8 | DataType::U8 | DataType::Bool => 1,
    };
    element_count * element_size
}

/// Get the alignment requirement for a data type.
pub fn get_alignment_requirement(dtype: DataType) -> usize {
    match dtype {
        DataType::F32 | DataType::I32 | DataType::U32 => 4,
        DataType::F16 | DataType::BF16 => 2,
        DataType::F64 | DataType::I64 => 8,
        DataType::I8 | DataType::U8 | DataType::Bool => 1,
    }
}

/// Get SIMD-friendly alignment (typically 32 bytes for AVX2, 64 bytes for AVX-512).
pub fn get_simd_alignment() -> usize {
    // Use 64-byte alignment for AVX-512 compatibility
    // This also works well for AVX2 (32 bytes) and provides cache line alignment
    64
}

/// System memory allocator using standard heap allocation.
#[derive(Debug)]
pub struct SystemMemoryAllocator {
    /// Memory usage statistics.
    stats: Arc<Mutex<AllocatorStats>>,
}

/// Memory allocator statistics.
#[derive(Debug, Default, Clone)]
pub struct AllocatorStats {
    /// Total bytes allocated.
    pub total_allocated: usize,
    /// Currently allocated bytes.
    pub current_allocated: usize,
    /// Peak allocated bytes.
    pub peak_allocated: usize,
    /// Number of allocations.
    pub allocation_count: usize,
    /// Number of deallocations.
    pub deallocation_count: usize,
}

impl Default for SystemMemoryAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemMemoryAllocator {
    /// Create a new system memory allocator.
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(AllocatorStats::default())),
        }
    }
}

impl TensorAllocator for SystemMemoryAllocator {
    fn allocate(&self, shape: &[usize], dtype: DataType) -> Result<TensorBuffer> {
        let size = calculate_tensor_size(shape, dtype);
        let alignment = get_alignment_requirement(dtype);

        if size == 0 {
            return Err(anyhow!("Cannot allocate zero-sized tensor"));
        }

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow!("Invalid memory layout: {}", e))?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(anyhow!("Memory allocation failed"));
            }
            raw_ptr
        };

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocated += size;
            stats.current_allocated += size;
            stats.peak_allocated = stats.peak_allocated.max(stats.current_allocated);
            stats.allocation_count += 1;
        }

        debug!(
            "Allocated {} bytes at {:?} with alignment {}",
            size, ptr, alignment
        );

        Ok(TensorBuffer {
            ptr,
            size,
            alignment,
            memory_type: MemoryType::SystemRAM,
        })
    }

    fn deallocate(&self, buffer: TensorBuffer) -> Result<()> {
        if buffer.size == 0 {
            return Ok(());
        }

        let layout = Layout::from_size_align(buffer.size, buffer.alignment)
            .map_err(|e| anyhow!("Invalid memory layout for deallocation: {}", e))?;

        unsafe {
            dealloc(buffer.ptr, layout);
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.current_allocated = stats.current_allocated.saturating_sub(buffer.size);
            stats.deallocation_count += 1;
        }

        debug!("Deallocated {} bytes at {:?}", buffer.size, buffer.ptr);

        Ok(())
    }

    fn get_memory_info(&self) -> MemoryInfo {
        let stats = self.stats.lock().unwrap();
        MemoryInfo {
            total_bytes: usize::MAX, // System memory limit is unknown
            allocated_bytes: stats.current_allocated,
            peak_bytes: stats.peak_allocated,
        }
    }
}

/// SIMD-aligned memory allocator for vectorized operations.
#[derive(Debug)]
pub struct AlignedMemoryAllocator {
    /// System allocator for actual allocation.
    system_allocator: SystemMemoryAllocator,
    /// Memory usage statistics.
    stats: Arc<Mutex<AllocatorStats>>,
}

impl Default for AlignedMemoryAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl AlignedMemoryAllocator {
    /// Create a new SIMD-aligned memory allocator.
    pub fn new() -> Self {
        Self {
            system_allocator: SystemMemoryAllocator::new(),
            stats: Arc::new(Mutex::new(AllocatorStats::default())),
        }
    }
}

impl TensorAllocator for AlignedMemoryAllocator {
    fn allocate(&self, shape: &[usize], dtype: DataType) -> Result<TensorBuffer> {
        let size = calculate_tensor_size(shape, dtype);
        let alignment = get_simd_alignment(); // Force SIMD alignment

        if size == 0 {
            return Err(anyhow!("Cannot allocate zero-sized tensor"));
        }

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow!("Invalid memory layout: {}", e))?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(anyhow!("Memory allocation failed"));
            }
            raw_ptr
        };

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocated += size;
            stats.current_allocated += size;
            stats.peak_allocated = stats.peak_allocated.max(stats.current_allocated);
            stats.allocation_count += 1;
        }

        debug!(
            "Allocated {} bytes at {:?} with SIMD alignment {}",
            size, ptr, alignment
        );

        Ok(TensorBuffer {
            ptr,
            size,
            alignment,
            memory_type: MemoryType::SystemRAM,
        })
    }

    fn deallocate(&self, buffer: TensorBuffer) -> Result<()> {
        if buffer.size == 0 {
            return Ok(());
        }

        let layout = Layout::from_size_align(buffer.size, buffer.alignment)
            .map_err(|e| anyhow!("Invalid memory layout for deallocation: {}", e))?;

        unsafe {
            dealloc(buffer.ptr, layout);
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.current_allocated = stats.current_allocated.saturating_sub(buffer.size);
            stats.deallocation_count += 1;
        }

        debug!("Deallocated {} bytes at {:?}", buffer.size, buffer.ptr);

        Ok(())
    }

    fn get_memory_info(&self) -> MemoryInfo {
        let stats = self.stats.lock().unwrap();
        MemoryInfo {
            total_bytes: usize::MAX,
            allocated_bytes: stats.current_allocated,
            peak_bytes: stats.peak_allocated,
        }
    }
}

/// Memory pool for efficient tensor allocation and reuse.
#[derive(Debug)]
pub struct PooledMemoryAllocator {
    /// Memory pools organized by size buckets.
    pools: RwLock<HashMap<usize, Vec<TensorBuffer>>>,
    /// System allocator for new allocations.
    system_allocator: SystemMemoryAllocator,
    /// Pool configuration.
    config: PoolConfig,
    /// Memory usage statistics.
    stats: Arc<Mutex<PoolStats>>,
}

/// Configuration for memory pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of buffers per size bucket.
    pub max_buffers_per_bucket: usize,
    /// Maximum total memory to keep in pools.
    pub max_pool_size: usize,
    /// Size bucket granularity (round up allocations to multiples of this).
    pub bucket_granularity: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_buffers_per_bucket: 64,
            max_pool_size: 256 * 1024 * 1024, // 256MB
            bucket_granularity: 64,           // 64-byte buckets for cache line alignment
        }
    }
}

/// Statistics from pooled allocator.
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    /// Allocator statistics.
    pub allocator_stats: AllocatorStats,
    /// Number of pool hits.
    pub pool_hits: usize,
    /// Number of pool misses.
    pub pool_misses: usize,
    /// Current pool size in bytes.
    pub pool_size: usize,
}

impl PooledMemoryAllocator {
    /// Create a new pooled memory allocator.
    pub fn new(config: PoolConfig) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            system_allocator: SystemMemoryAllocator::new(),
            config,
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Round size up to the nearest bucket boundary.
    fn round_to_bucket(&self, size: usize) -> usize {
        let granularity = self.config.bucket_granularity;
        ((size + granularity - 1) / granularity) * granularity
    }

    /// Try to get a buffer from the pool.
    fn try_get_from_pool(&self, bucket_size: usize, alignment: usize) -> Option<TensorBuffer> {
        let mut pools = self.pools.write().unwrap();

        if let Some(buffers) = pools.get_mut(&bucket_size) {
            // Find a buffer with compatible alignment
            for i in 0..buffers.len() {
                if buffers[i].alignment >= alignment {
                    let buffer = buffers.swap_remove(i);

                    // Update statistics
                    {
                        let mut stats = self.stats.lock().unwrap();
                        stats.pool_hits += 1;
                        stats.pool_size -= buffer.size;
                    }

                    debug!("Pool hit: reusing buffer of size {} bytes", bucket_size);
                    return Some(buffer);
                }
            }
        }

        {
            let mut stats = self.stats.lock().unwrap();
            stats.pool_misses += 1;
        }

        None
    }

    /// Return a buffer to the pool.
    fn return_to_pool(&self, buffer: TensorBuffer) -> bool {
        let bucket_size = self.round_to_bucket(buffer.size);
        let mut pools = self.pools.write().unwrap();

        let buffers = pools.entry(bucket_size).or_insert_with(Vec::new);

        // Check pool limits
        if buffers.len() >= self.config.max_buffers_per_bucket {
            return false; // Pool bucket is full
        }

        let stats = self.stats.lock().unwrap();
        if stats.pool_size + buffer.size > self.config.max_pool_size {
            return false; // Pool is full
        }
        drop(stats);

        let buffer_size = buffer.size;
        buffers.push(buffer);

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.pool_size += buffer_size;
        }

        debug!("Returned buffer of size {} bytes to pool", buffer_size);
        true
    }
}

impl TensorAllocator for PooledMemoryAllocator {
    fn allocate(&self, shape: &[usize], dtype: DataType) -> Result<TensorBuffer> {
        let size = calculate_tensor_size(shape, dtype);
        let alignment = get_simd_alignment(); // Use SIMD alignment for better performance
        let bucket_size = self.round_to_bucket(size);

        if size == 0 {
            return Err(anyhow!("Cannot allocate zero-sized tensor"));
        }

        // Try to get from pool first
        if let Some(mut buffer) = self.try_get_from_pool(bucket_size, alignment) {
            // Adjust the buffer size to requested size (bucket size might be larger)
            buffer.size = size;
            return Ok(buffer);
        }

        // Allocate new buffer
        let layout = Layout::from_size_align(bucket_size, alignment)
            .map_err(|e| anyhow!("Invalid memory layout: {}", e))?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(anyhow!("Memory allocation failed"));
            }
            raw_ptr
        };

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.allocator_stats.total_allocated += bucket_size;
            stats.allocator_stats.current_allocated += bucket_size;
            stats.allocator_stats.peak_allocated = stats
                .allocator_stats
                .peak_allocated
                .max(stats.allocator_stats.current_allocated);
            stats.allocator_stats.allocation_count += 1;
        }

        debug!("Allocated new buffer: {} bytes at {:?}", bucket_size, ptr);

        Ok(TensorBuffer {
            ptr,
            size, // Return requested size, not bucket size
            alignment,
            memory_type: MemoryType::SystemRAM,
        })
    }

    fn deallocate(&self, mut buffer: TensorBuffer) -> Result<()> {
        if buffer.size == 0 {
            return Ok(());
        }

        // Restore bucket size for pool management
        let bucket_size = self.round_to_bucket(buffer.size);
        let buffer_ptr = buffer.ptr;
        let buffer_alignment = buffer.alignment;
        buffer.size = bucket_size;

        // Try to return to pool
        if self.return_to_pool(buffer) {
            return Ok(()); // Successfully returned to pool
        }

        // Pool is full, deallocate immediately
        let layout = Layout::from_size_align(bucket_size, buffer_alignment)
            .map_err(|e| anyhow!("Invalid memory layout for deallocation: {}", e))?;

        unsafe {
            dealloc(buffer_ptr, layout);
        }

        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.allocator_stats.current_allocated = stats
                .allocator_stats
                .current_allocated
                .saturating_sub(bucket_size);
            stats.allocator_stats.deallocation_count += 1;
        }

        debug!(
            "Deallocated buffer: {} bytes at {:?}",
            bucket_size, buffer_ptr as *const u8
        );

        Ok(())
    }

    fn get_memory_info(&self) -> MemoryInfo {
        let stats = self.stats.lock().unwrap();
        MemoryInfo {
            total_bytes: usize::MAX,
            allocated_bytes: stats.allocator_stats.current_allocated,
            peak_bytes: stats.allocator_stats.peak_allocated,
        }
    }
}

impl Drop for PooledMemoryAllocator {
    fn drop(&mut self) {
        // Clean up all pooled buffers
        let mut pools = self.pools.write().unwrap();
        for (bucket_size, buffers) in pools.drain() {
            let buffer_count = buffers.len();
            for buffer in buffers {
                let layout = Layout::from_size_align(buffer.size, buffer.alignment).unwrap();
                unsafe {
                    dealloc(buffer.ptr, layout);
                }
            }
            debug!(
                "Cleaned up {} buffers from bucket {}",
                buffer_count, bucket_size
            );
        }
    }
}

/// Get statistics from a pooled allocator.
impl PooledMemoryAllocator {
    /// Get detailed pool statistics.
    pub fn get_pool_stats(&self) -> PoolStats {
        let stats = self.stats.lock().unwrap();
        PoolStats {
            allocator_stats: AllocatorStats {
                total_allocated: stats.allocator_stats.total_allocated,
                current_allocated: stats.allocator_stats.current_allocated,
                peak_allocated: stats.allocator_stats.peak_allocated,
                allocation_count: stats.allocator_stats.allocation_count,
                deallocation_count: stats.allocator_stats.deallocation_count,
            },
            pool_hits: stats.pool_hits,
            pool_misses: stats.pool_misses,
            pool_size: stats.pool_size,
        }
    }

    /// Get pool hit rate.
    pub fn get_hit_rate(&self) -> f64 {
        let stats = self.stats.lock().unwrap();
        let total_requests = stats.pool_hits + stats.pool_misses;
        if total_requests > 0 {
            stats.pool_hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }

    /// Clear all pools.
    pub fn clear_pools(&self) {
        let mut pools = self.pools.write().unwrap();
        for (_, buffers) in pools.drain() {
            for buffer in buffers {
                let layout = Layout::from_size_align(buffer.size, buffer.alignment).unwrap();
                unsafe {
                    dealloc(buffer.ptr, layout);
                }
            }
        }

        let mut stats = self.stats.lock().unwrap();
        stats.pool_size = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_size_calculation() {
        assert_eq!(calculate_tensor_size(&[2, 3], DataType::F32), 24); // 2*3*4 = 24
        assert_eq!(calculate_tensor_size(&[4], DataType::F16), 8); // 4*2 = 8
        assert_eq!(calculate_tensor_size(&[10], DataType::U8), 10); // 10*1 = 10
    }

    #[test]
    fn test_system_allocator() -> Result<()> {
        let allocator = SystemMemoryAllocator::new();

        let buffer = allocator.allocate(&[10], DataType::F32)?;
        assert_eq!(buffer.size, 40);
        assert_eq!(buffer.alignment, 4);
        assert_eq!(buffer.memory_type, MemoryType::SystemRAM);

        let memory_info = allocator.get_memory_info();
        assert_eq!(memory_info.allocated_bytes, 40);

        allocator.deallocate(buffer)?;

        let memory_info = allocator.get_memory_info();
        assert_eq!(memory_info.allocated_bytes, 0);

        Ok(())
    }

    #[test]
    fn test_aligned_allocator() -> Result<()> {
        let allocator = AlignedMemoryAllocator::new();

        let buffer = allocator.allocate(&[16], DataType::F32)?;
        assert_eq!(buffer.size, 64);
        assert_eq!(buffer.alignment, 64); // SIMD alignment

        // Check that pointer is properly aligned
        assert_eq!(buffer.ptr as usize % buffer.alignment, 0);

        allocator.deallocate(buffer)?;

        Ok(())
    }

    #[test]
    fn test_pooled_allocator() -> Result<()> {
        let config = PoolConfig {
            max_buffers_per_bucket: 2,
            max_pool_size: 1024,
            bucket_granularity: 64,
        };
        let allocator = PooledMemoryAllocator::new(config);

        // First allocation should be a pool miss
        let buffer1 = allocator.allocate(&[10], DataType::F32)?;
        let stats = allocator.get_pool_stats();
        assert_eq!(stats.pool_misses, 1);
        assert_eq!(stats.pool_hits, 0);

        // Return to pool
        allocator.deallocate(buffer1)?;
        let stats = allocator.get_pool_stats();
        assert!(stats.pool_size > 0);

        // Second allocation should be a pool hit
        let buffer2 = allocator.allocate(&[10], DataType::F32)?;
        let stats = allocator.get_pool_stats();
        assert_eq!(stats.pool_hits, 1);

        allocator.deallocate(buffer2)?;

        // Check hit rate
        let hit_rate = allocator.get_hit_rate();
        assert_eq!(hit_rate, 0.5); // 1 hit out of 2 attempts

        Ok(())
    }

    #[test]
    fn test_pool_limits() -> Result<()> {
        let config = PoolConfig {
            max_buffers_per_bucket: 1, // Only one buffer per bucket
            max_pool_size: 128,        // Small pool size
            bucket_granularity: 64,
        };
        let allocator = PooledMemoryAllocator::new(config);

        // Allocate and deallocate first buffer (should go to pool)
        let buffer1 = allocator.allocate(&[10], DataType::F32)?;
        allocator.deallocate(buffer1)?;

        // Allocate and deallocate second buffer (should be rejected by pool)
        let buffer2 = allocator.allocate(&[10], DataType::F32)?;
        allocator.deallocate(buffer2)?;

        let stats = allocator.get_pool_stats();

        // Only the first buffer should be in the pool due to max_buffers_per_bucket limit
        assert!(stats.pool_size <= 128);

        Ok(())
    }
}
