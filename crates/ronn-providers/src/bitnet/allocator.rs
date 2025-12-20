//! Memory allocator optimized for BitNet bit-packed tensors.
//!
//! This allocator is specialized for managing memory for bit-packed tensors,
//! providing efficient allocation and deallocation with proper alignment
//! for SIMD operations on packed bit data.

use std::alloc::{Layout, alloc, dealloc};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use ronn_core::{DataType, MemoryInfo, MemoryType, TensorAllocator, TensorBuffer};

/// Memory allocator optimized for BitNet bit-packed tensors.
#[derive(Debug)]
pub struct BitNetMemoryAllocator {
    /// Total allocated bytes.
    allocated_bytes: Arc<AtomicUsize>,
    /// Peak memory usage.
    peak_bytes: Arc<AtomicUsize>,
    /// Total available system memory (estimated).
    total_bytes: usize,
}

impl BitNetMemoryAllocator {
    /// Create a new BitNet memory allocator.
    pub fn new() -> Self {
        // Estimate total system memory (simplified)
        let total_bytes = 8 * 1024 * 1024 * 1024; // Assume 8GB for now

        Self {
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            peak_bytes: Arc::new(AtomicUsize::new(0)),
            total_bytes,
        }
    }

    /// Calculate the memory size needed for bit-packed tensor.
    fn calculate_packed_size(&self, shape: &[usize], dtype: DataType) -> Result<usize> {
        let element_count: usize = shape.iter().product();

        match dtype {
            DataType::Bool => {
                // 1 bit per element for binary quantization
                Ok((element_count + 7) / 8) // Round up to nearest byte
            }
            DataType::U8 => {
                // Use U8 to represent 2-bit ternary (4 elements per byte)
                Ok((element_count + 3) / 4) // Round up to nearest 4 elements
            }
            DataType::F32 | DataType::F16 | DataType::BF16 => {
                // Standard tensor allocation (for activations)
                let element_size = match dtype {
                    DataType::F32 => std::mem::size_of::<f32>(),
                    DataType::F16 | DataType::BF16 => std::mem::size_of::<u16>(),
                    _ => unreachable!(),
                };
                Ok(element_count * element_size)
            }
            _ => Err(anyhow!(
                "Unsupported data type for BitNet allocator: {:?}",
                dtype
            )),
        }
    }

    /// Get optimal alignment for bit-packed data.
    fn get_alignment(&self, dtype: DataType) -> usize {
        match dtype {
            DataType::Bool | DataType::U8 => 32, // 32-byte alignment for SIMD
            DataType::F32 => 32,                 // 32-byte alignment for AVX2
            DataType::F16 | DataType::BF16 => 16, // 16-byte alignment for SSE
            _ => 8,                              // Default alignment
        }
    }

    /// Update memory statistics.
    fn update_stats(&self, size: usize, allocating: bool) {
        if allocating {
            let new_allocated = self.allocated_bytes.fetch_add(size, Ordering::Relaxed) + size;

            // Update peak if necessary
            let current_peak = self.peak_bytes.load(Ordering::Relaxed);
            if new_allocated > current_peak {
                self.peak_bytes.store(new_allocated, Ordering::Relaxed);
            }
        } else {
            self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }
}

impl Default for BitNetMemoryAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorAllocator for BitNetMemoryAllocator {
    fn allocate(&self, shape: &[usize], dtype: DataType) -> Result<TensorBuffer> {
        let size = self.calculate_packed_size(shape, dtype)?;
        let alignment = self.get_alignment(dtype);

        // Create memory layout
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow!("Invalid memory layout: {}", e))?;

        // Allocate aligned memory
        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(anyhow!("Memory allocation failed for {} bytes", size));
            }
            raw_ptr
        };

        // Update statistics
        self.update_stats(size, true);

        Ok(TensorBuffer {
            ptr,
            size,
            alignment,
            memory_type: MemoryType::SystemRAM,
        })
    }

    fn deallocate(&self, buffer: TensorBuffer) -> Result<()> {
        let layout = Layout::from_size_align(buffer.size, buffer.alignment)
            .map_err(|e| anyhow!("Invalid memory layout for deallocation: {}", e))?;

        unsafe {
            dealloc(buffer.ptr, layout);
        }

        // Update statistics
        self.update_stats(buffer.size, false);

        Ok(())
    }

    fn get_memory_info(&self) -> MemoryInfo {
        MemoryInfo {
            total_bytes: self.total_bytes,
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Create a BitNet memory allocator.
pub fn create_bitnet_allocator() -> Arc<dyn TensorAllocator> {
    Arc::new(BitNetMemoryAllocator::new())
}

/// Statistics specific to BitNet memory usage.
#[derive(Debug, Clone)]
pub struct BitNetMemoryStats {
    /// Number of binary tensors allocated.
    pub binary_tensor_count: usize,
    /// Number of ternary tensors allocated.
    pub ternary_tensor_count: usize,
    /// Total memory saved compared to FP32.
    pub memory_saved_bytes: usize,
    /// Average compression ratio achieved.
    pub average_compression_ratio: f32,
}

impl BitNetMemoryStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self {
            binary_tensor_count: 0,
            ternary_tensor_count: 0,
            memory_saved_bytes: 0,
            average_compression_ratio: 1.0,
        }
    }

    /// Update statistics with a new binary tensor allocation.
    pub fn add_binary_tensor(&mut self, original_size: usize, compressed_size: usize) {
        self.binary_tensor_count += 1;
        self.memory_saved_bytes += original_size.saturating_sub(compressed_size);
        self.recalculate_compression_ratio();
    }

    /// Update statistics with a new ternary tensor allocation.
    pub fn add_ternary_tensor(&mut self, original_size: usize, compressed_size: usize) {
        self.ternary_tensor_count += 1;
        self.memory_saved_bytes += original_size.saturating_sub(compressed_size);
        self.recalculate_compression_ratio();
    }

    fn recalculate_compression_ratio(&mut self) {
        let total_tensors = self.binary_tensor_count + self.ternary_tensor_count;
        if total_tensors > 0 {
            // Estimate based on typical compression ratios
            let binary_contribution = self.binary_tensor_count as f32 * 32.0; // ~32x compression
            let ternary_contribution = self.ternary_tensor_count as f32 * 16.0; // ~16x compression

            self.average_compression_ratio =
                (binary_contribution + ternary_contribution) / total_tensors as f32;
        }
    }
}

impl Default for BitNetMemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitnet_allocator_creation() {
        let allocator = BitNetMemoryAllocator::new();
        let memory_info = allocator.get_memory_info();

        assert_eq!(memory_info.allocated_bytes, 0);
        assert_eq!(memory_info.peak_bytes, 0);
        assert!(memory_info.total_bytes > 0);
    }

    #[test]
    fn test_binary_tensor_allocation() -> Result<()> {
        let allocator = BitNetMemoryAllocator::new();

        // Allocate space for 1000 binary elements (should need 125 bytes)
        let buffer = allocator.allocate(&[1000], DataType::Bool)?;

        assert_eq!(buffer.size, 125); // (1000 + 7) / 8 = 125 bytes
        assert_eq!(buffer.alignment, 32); // SIMD alignment
        assert_eq!(buffer.memory_type, MemoryType::SystemRAM);

        // Check memory stats
        let memory_info = allocator.get_memory_info();
        assert_eq!(memory_info.allocated_bytes, 125);
        assert_eq!(memory_info.peak_bytes, 125);

        // Deallocate
        allocator.deallocate(buffer)?;

        let memory_info_after = allocator.get_memory_info();
        assert_eq!(memory_info_after.allocated_bytes, 0);
        assert_eq!(memory_info_after.peak_bytes, 125); // Peak remains

        Ok(())
    }

    #[test]
    fn test_ternary_tensor_allocation() -> Result<()> {
        let allocator = BitNetMemoryAllocator::new();

        // Allocate space for 100 ternary elements (should need 25 bytes)
        let buffer = allocator.allocate(&[100], DataType::U8)?;

        assert_eq!(buffer.size, 25); // (100 + 3) / 4 = 25 bytes
        assert_eq!(buffer.alignment, 32);

        allocator.deallocate(buffer)?;
        Ok(())
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = BitNetMemoryStats::new();

        // Add some tensor allocations
        stats.add_binary_tensor(4000, 125); // 1000 FP32 -> 125 bytes
        stats.add_ternary_tensor(1600, 100); // 400 FP32 -> 100 bytes

        assert_eq!(stats.binary_tensor_count, 1);
        assert_eq!(stats.ternary_tensor_count, 1);
        assert_eq!(stats.memory_saved_bytes, 4000 - 125 + 1600 - 100);
        assert!(stats.average_compression_ratio > 20.0); // Should be around 24x
    }

    #[test]
    fn test_multi_dimensional_allocation() -> Result<()> {
        let allocator = BitNetMemoryAllocator::new();

        // Allocate 2D binary tensor: 32x32 = 1024 elements
        let buffer = allocator.allocate(&[32, 32], DataType::Bool)?;

        assert_eq!(buffer.size, 128); // (1024 + 7) / 8 = 128 bytes

        allocator.deallocate(buffer)?;
        Ok(())
    }
}
