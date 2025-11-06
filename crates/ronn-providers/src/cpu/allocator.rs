//! CPU-specific memory allocator with NUMA awareness.
//!
//! This module provides CPU-optimized memory allocation strategies including
//! NUMA-aware allocation and cache-friendly memory layouts.

use std::sync::Arc;

use anyhow::Result;
use ronn_core::{DataType, MemoryInfo, TensorAllocator, TensorBuffer};

use super::simd::{detect_simd_capabilities, get_optimal_vector_width, SimdCapabilities};
use crate::allocator::{AlignedMemoryAllocator, PoolConfig, PooledMemoryAllocator};

/// CPU-specific memory allocator with SIMD alignment and optional NUMA awareness.
#[derive(Debug)]
pub struct CpuMemoryAllocator {
    /// Underlying pooled allocator for memory reuse.
    pooled_allocator: PooledMemoryAllocator,
    /// Fallback aligned allocator.
    aligned_allocator: AlignedMemoryAllocator,
    /// SIMD capabilities for optimal alignment.
    simd_capabilities: SimdCapabilities,
    /// NUMA node preference (-1 for no preference).
    numa_node: i32,
}

impl Default for CpuMemoryAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuMemoryAllocator {
    /// Create a new CPU memory allocator with default configuration.
    pub fn new() -> Self {
        let simd_capabilities = detect_simd_capabilities();
        let vector_width = get_optimal_vector_width(&simd_capabilities);

        let pool_config = PoolConfig {
            max_buffers_per_bucket: 128,      // More aggressive pooling for CPU
            max_pool_size: 512 * 1024 * 1024, // 512MB pool
            bucket_granularity: vector_width, // Align to SIMD width
        };

        Self {
            pooled_allocator: PooledMemoryAllocator::new(pool_config),
            aligned_allocator: AlignedMemoryAllocator::new(),
            simd_capabilities,
            numa_node: -1, // No NUMA preference by default
        }
    }

    /// Create a CPU memory allocator with custom pool configuration.
    pub fn with_config(config: PoolConfig) -> Self {
        let simd_capabilities = detect_simd_capabilities();

        Self {
            pooled_allocator: PooledMemoryAllocator::new(config),
            aligned_allocator: AlignedMemoryAllocator::new(),
            simd_capabilities,
            numa_node: -1,
        }
    }

    /// Create a NUMA-aware CPU memory allocator.
    pub fn with_numa_node(numa_node: i32) -> Self {
        let mut allocator = Self::new();
        allocator.numa_node = numa_node;
        allocator
    }

    /// Get the SIMD capabilities of this allocator.
    pub fn get_simd_capabilities(&self) -> &SimdCapabilities {
        &self.simd_capabilities
    }

    /// Get the preferred NUMA node.
    pub fn get_numa_node(&self) -> i32 {
        self.numa_node
    }

    /// Set the preferred NUMA node.
    pub fn set_numa_node(&mut self, numa_node: i32) {
        self.numa_node = numa_node;
    }

    /// Check if memory should use pooling based on size and usage pattern.
    fn should_use_pooling(&self, size: usize) -> bool {
        // Use pooling for medium-sized allocations
        // Very small allocations have low overhead anyway
        // Very large allocations might fragment the pool
        const MIN_POOL_SIZE: usize = 1024; // 1KB
        const MAX_POOL_SIZE: usize = 16 * 1024 * 1024; // 16MB

        size >= MIN_POOL_SIZE && size <= MAX_POOL_SIZE
    }

    /// Allocate memory on a specific NUMA node (Linux-specific).
    #[cfg(target_os = "linux")]
    fn allocate_numa(&self, buffer: TensorBuffer) -> Result<TensorBuffer> {
        if self.numa_node < 0 {
            return Ok(buffer); // No NUMA preference
        }

        // In a real implementation, we would use libnuma or similar
        // to bind memory to specific NUMA nodes
        // For now, just return the buffer as-is
        Ok(buffer)
    }

    #[cfg(not(target_os = "linux"))]
    fn allocate_numa(&self, buffer: TensorBuffer) -> Result<TensorBuffer> {
        // NUMA support is Linux-specific for now
        Ok(buffer)
    }

    /// Get detailed allocation statistics.
    pub fn get_detailed_stats(&self) -> CpuAllocatorStats {
        let memory_info = self.get_memory_info();

        CpuAllocatorStats {
            memory_info: memory_info.clone(),
            pool_hit_rate: self.pooled_allocator.get_hit_rate(),
            pool_size: memory_info.allocated_bytes, // Use allocated_bytes as proxy for pool size
            numa_node: self.numa_node,
            simd_alignment: get_optimal_vector_width(&self.simd_capabilities),
        }
    }

    /// Clear all memory pools.
    pub fn clear_pools(&self) {
        self.pooled_allocator.clear_pools();
    }
}

impl TensorAllocator for CpuMemoryAllocator {
    fn allocate(&self, shape: &[usize], dtype: DataType) -> Result<TensorBuffer> {
        let size = crate::allocator::calculate_tensor_size(shape, dtype);

        let buffer = if self.should_use_pooling(size) {
            // Use pooled allocation for medium-sized tensors
            self.pooled_allocator.allocate(shape, dtype)?
        } else {
            // Use aligned allocation for other sizes
            self.aligned_allocator.allocate(shape, dtype)?
        };

        // Apply NUMA placement if requested
        let numa_buffer = self.allocate_numa(buffer)?;

        Ok(numa_buffer)
    }

    fn deallocate(&self, buffer: TensorBuffer) -> Result<()> {
        let size = buffer.size;

        if self.should_use_pooling(size) {
            self.pooled_allocator.deallocate(buffer)
        } else {
            self.aligned_allocator.deallocate(buffer)
        }
    }

    fn get_memory_info(&self) -> MemoryInfo {
        // Combine statistics from both allocators
        let pool_info = self.pooled_allocator.get_memory_info();
        let aligned_info = self.aligned_allocator.get_memory_info();

        MemoryInfo {
            total_bytes: pool_info.total_bytes,
            allocated_bytes: pool_info.allocated_bytes + aligned_info.allocated_bytes,
            peak_bytes: pool_info.peak_bytes.max(aligned_info.peak_bytes),
        }
    }
}

/// Detailed CPU allocator statistics.
#[derive(Debug, Clone)]
pub struct CpuAllocatorStats {
    /// Basic memory information.
    pub memory_info: MemoryInfo,
    /// Pool hit rate (0.0 to 1.0).
    pub pool_hit_rate: f64,
    /// Current pool size in bytes.
    pub pool_size: usize,
    /// NUMA node preference (-1 for none).
    pub numa_node: i32,
    /// SIMD alignment in bytes.
    pub simd_alignment: usize,
}

/// Create a shared CPU allocator instance.
pub fn create_cpu_allocator() -> Arc<dyn TensorAllocator> {
    Arc::new(CpuMemoryAllocator::new())
}

/// Create a NUMA-aware CPU allocator instance.
pub fn create_numa_cpu_allocator(numa_node: i32) -> Arc<dyn TensorAllocator> {
    Arc::new(CpuMemoryAllocator::with_numa_node(numa_node))
}

/// Create a CPU allocator with custom pool configuration.
pub fn create_cpu_allocator_with_config(config: PoolConfig) -> Arc<dyn TensorAllocator> {
    Arc::new(CpuMemoryAllocator::with_config(config))
}

/// Detect the number of NUMA nodes on the system (Linux-specific).
#[cfg(target_os = "linux")]
pub fn detect_numa_nodes() -> usize {
    // In a real implementation, we would read from /sys/devices/system/node/
    // or use libnuma to detect the number of NUMA nodes
    // For now, return 1 (single node)
    1
}

/// Detect the number of NUMA nodes on the system (non-Linux fallback).
#[cfg(not(target_os = "linux"))]
pub fn detect_numa_nodes() -> usize {
    1 // Non-Linux systems assumed to have single NUMA node
}

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::MemoryType;

    #[test]
    fn test_cpu_allocator_creation() {
        let allocator = CpuMemoryAllocator::new();
        let capabilities = allocator.get_simd_capabilities();

        // Should detect some capabilities on most systems
        println!("SIMD capabilities: {:?}", capabilities);

        assert_eq!(allocator.get_numa_node(), -1);
    }

    #[test]
    fn test_numa_configuration() {
        let mut allocator = CpuMemoryAllocator::new();
        assert_eq!(allocator.get_numa_node(), -1);

        allocator.set_numa_node(0);
        assert_eq!(allocator.get_numa_node(), 0);
    }

    #[test]
    fn test_pooling_decision() -> Result<()> {
        let allocator = CpuMemoryAllocator::new();

        // Small allocation should not use pooling
        let small_size = 512; // 512 bytes
        assert!(!allocator.should_use_pooling(small_size));

        // Medium allocation should use pooling
        let medium_size = 64 * 1024; // 64KB
        assert!(allocator.should_use_pooling(medium_size));

        // Large allocation should not use pooling
        let large_size = 32 * 1024 * 1024; // 32MB
        assert!(!allocator.should_use_pooling(large_size));

        Ok(())
    }

    #[test]
    fn test_allocation_and_deallocation() -> Result<()> {
        let allocator = CpuMemoryAllocator::new();

        // Test medium-sized allocation (should use pooling)
        let buffer = allocator.allocate(&[1024], DataType::F32)?;
        assert_eq!(buffer.size, 4096); // 1024 * 4 bytes
        assert!(buffer.alignment >= 4);
        assert_eq!(buffer.memory_type, MemoryType::SystemRAM);

        let stats_before_dealloc = allocator.get_detailed_stats();
        assert!(stats_before_dealloc.memory_info.allocated_bytes > 0);

        allocator.deallocate(buffer)?;

        Ok(())
    }

    #[test]
    fn test_pool_statistics() -> Result<()> {
        let allocator = CpuMemoryAllocator::new();

        // Allocate and deallocate to test pooling
        let buffer1 = allocator.allocate(&[256], DataType::F32)?; // 1KB, should use pooling
        allocator.deallocate(buffer1)?;

        let buffer2 = allocator.allocate(&[256], DataType::F32)?; // Should reuse from pool
        let stats = allocator.get_detailed_stats();

        // Should have some pool activity
        println!("Pool hit rate: {}", stats.pool_hit_rate);

        allocator.deallocate(buffer2)?;

        Ok(())
    }

    #[test]
    fn test_shared_allocator_creation() {
        let allocator = create_cpu_allocator();
        let memory_info = allocator.get_memory_info();

        // Should start with no allocated memory
        assert_eq!(memory_info.allocated_bytes, 0);
    }

    #[test]
    fn test_numa_detection() {
        let numa_nodes = detect_numa_nodes();
        assert!(numa_nodes >= 1); // Should detect at least one node
        println!("Detected {} NUMA nodes", numa_nodes);
    }

    #[test]
    fn test_custom_config() {
        let config = PoolConfig {
            max_buffers_per_bucket: 16,
            max_pool_size: 1024 * 1024, // 1MB
            bucket_granularity: 32,
        };

        let allocator = CpuMemoryAllocator::with_config(config);
        let stats = allocator.get_detailed_stats();

        assert_eq!(
            stats.simd_alignment,
            get_optimal_vector_width(allocator.get_simd_capabilities())
        );
    }
}
