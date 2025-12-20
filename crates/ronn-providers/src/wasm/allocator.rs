//! WebAssembly memory allocator for linear memory management.
//!
//! This allocator is optimized for WebAssembly's linear memory model,
//! providing efficient allocation and deallocation with proper alignment
//! for WASM SIMD operations and JavaScript TypedArray integration.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow};
use ronn_core::{DataType, MemoryInfo, MemoryType, TensorAllocator, TensorBuffer};

/// Memory allocator optimized for WebAssembly linear memory.
#[derive(Debug)]
pub struct WasmMemoryAllocator {
    /// Current allocated bytes.
    allocated_bytes: Arc<AtomicUsize>,
    /// Peak memory usage.
    peak_bytes: Arc<AtomicUsize>,
    /// Maximum available WebAssembly memory.
    max_memory_bytes: usize,
    /// Page size (64KB in WASM).
    page_size: usize,
}

impl WasmMemoryAllocator {
    /// Create a new WASM memory allocator.
    pub fn new() -> Self {
        // WebAssembly linear memory constraints
        let max_memory_bytes = 2 * 1024 * 1024 * 1024; // 2GB theoretical max
        let page_size = 64 * 1024; // 64KB per WASM memory page

        Self {
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            peak_bytes: Arc::new(AtomicUsize::new(0)),
            max_memory_bytes,
            page_size,
        }
    }

    /// Create a WASM allocator with custom memory limit.
    pub fn with_memory_limit(memory_limit: usize) -> Self {
        let page_size = 64 * 1024;

        // Align memory limit to page boundaries
        let aligned_limit = ((memory_limit + page_size - 1) / page_size) * page_size;

        Self {
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            peak_bytes: Arc::new(AtomicUsize::new(0)),
            max_memory_bytes: aligned_limit,
            page_size,
        }
    }

    /// Calculate tensor memory requirements.
    fn calculate_tensor_size(&self, shape: &[usize], dtype: DataType) -> usize {
        let element_count: usize = shape.iter().product();
        let element_size = match dtype {
            DataType::F32 => std::mem::size_of::<f32>(),
            DataType::F16 | DataType::BF16 => std::mem::size_of::<u16>(),
            DataType::F64 => std::mem::size_of::<f64>(),
            DataType::U8 => std::mem::size_of::<u8>(),
            DataType::I8 => std::mem::size_of::<i8>(),
            DataType::I32 => std::mem::size_of::<i32>(),
            DataType::I64 => std::mem::size_of::<i64>(),
            DataType::U32 => std::mem::size_of::<u32>(),
            DataType::Bool => std::mem::size_of::<u8>(), // Represented as u8 in WASM
        };

        element_count * element_size
    }

    /// Get alignment for WASM SIMD operations.
    fn get_wasm_alignment(&self, dtype: DataType) -> usize {
        match dtype {
            DataType::F32 | DataType::I32 | DataType::U32 => 16, // 128-bit SIMD alignment
            DataType::F16 | DataType::BF16 => 8,                 // Half precision
            DataType::F64 | DataType::I64 => 16,                 // 64-bit types
            DataType::U8 | DataType::I8 | DataType::Bool => 16,  // Byte types with SIMD
        }
    }

    /// Check if allocation would exceed memory limits.
    fn check_memory_limit(&self, size: usize) -> Result<()> {
        let current_allocated = self.allocated_bytes.load(Ordering::Relaxed);
        let total_after_allocation = current_allocated + size;

        if total_after_allocation > self.max_memory_bytes {
            return Err(anyhow!(
                "Allocation would exceed WASM memory limit: {} + {} > {}",
                current_allocated,
                size,
                self.max_memory_bytes
            ));
        }

        Ok(())
    }

    /// Update memory usage statistics.
    fn update_stats(&self, size: usize, allocating: bool) {
        if allocating {
            let new_allocated = self.allocated_bytes.fetch_add(size, Ordering::Relaxed) + size;

            // Update peak memory usage
            let current_peak = self.peak_bytes.load(Ordering::Relaxed);
            if new_allocated > current_peak {
                self.peak_bytes.store(new_allocated, Ordering::Relaxed);
            }
        } else {
            self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
        }
    }

    /// Allocate WASM-compatible memory.
    #[cfg(target_arch = "wasm32")]
    fn allocate_wasm_memory(&self, size: usize, alignment: usize) -> Result<*mut u8> {
        use std::alloc::{Layout, alloc};

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow!("Invalid memory layout: {}", e))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow!("Failed to allocate {} bytes in WASM memory", size));
        }

        Ok(ptr)
    }

    /// Allocate memory (fallback for non-WASM targets).
    #[cfg(not(target_arch = "wasm32"))]
    fn allocate_wasm_memory(&self, size: usize, alignment: usize) -> Result<*mut u8> {
        use std::alloc::{Layout, alloc};

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow!("Invalid memory layout: {}", e))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(anyhow!("Failed to allocate {} bytes", size));
        }

        Ok(ptr)
    }

    /// Deallocate WASM memory.
    #[cfg(target_arch = "wasm32")]
    fn deallocate_wasm_memory(&self, ptr: *mut u8, size: usize, alignment: usize) -> Result<()> {
        use std::alloc::{Layout, dealloc};

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow!("Invalid memory layout for deallocation: {}", e))?;

        unsafe {
            dealloc(ptr, layout);
        }

        Ok(())
    }

    /// Deallocate memory (fallback for non-WASM targets).
    #[cfg(not(target_arch = "wasm32"))]
    fn deallocate_wasm_memory(&self, ptr: *mut u8, size: usize, alignment: usize) -> Result<()> {
        use std::alloc::{Layout, dealloc};

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| anyhow!("Invalid memory layout for deallocation: {}", e))?;

        unsafe {
            dealloc(ptr, layout);
        }

        Ok(())
    }

    /// Get available memory pages.
    pub fn get_available_pages(&self) -> usize {
        let current_allocated = self.allocated_bytes.load(Ordering::Relaxed);
        let available_bytes = self.max_memory_bytes.saturating_sub(current_allocated);
        available_bytes / self.page_size
    }

    /// Get memory utilization percentage.
    pub fn get_memory_utilization(&self) -> f32 {
        let current_allocated = self.allocated_bytes.load(Ordering::Relaxed);
        (current_allocated as f32 / self.max_memory_bytes as f32) * 100.0
    }
}

impl Default for WasmMemoryAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorAllocator for WasmMemoryAllocator {
    fn allocate(&self, shape: &[usize], dtype: DataType) -> Result<TensorBuffer> {
        let size = self.calculate_tensor_size(shape, dtype);
        let alignment = self.get_wasm_alignment(dtype);

        // Check memory limits before allocation
        self.check_memory_limit(size)?;

        // Allocate WASM-compatible memory
        let ptr = self.allocate_wasm_memory(size, alignment)?;

        // Update statistics
        self.update_stats(size, true);

        Ok(TensorBuffer {
            ptr,
            size,
            alignment,
            memory_type: MemoryType::SystemRAM, // WASM linear memory
        })
    }

    fn deallocate(&self, buffer: TensorBuffer) -> Result<()> {
        // Deallocate the memory
        self.deallocate_wasm_memory(buffer.ptr, buffer.size, buffer.alignment)?;

        // Update statistics
        self.update_stats(buffer.size, false);

        Ok(())
    }

    fn get_memory_info(&self) -> MemoryInfo {
        MemoryInfo {
            total_bytes: self.max_memory_bytes,
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Create a WASM memory allocator with default settings.
pub fn create_wasm_allocator() -> Arc<dyn TensorAllocator> {
    Arc::new(WasmMemoryAllocator::new())
}

/// Create a WASM memory allocator with custom memory limit.
pub fn create_wasm_allocator_with_limit(memory_limit: usize) -> Arc<dyn TensorAllocator> {
    Arc::new(WasmMemoryAllocator::with_memory_limit(memory_limit))
}

/// WASM-specific memory statistics.
#[derive(Debug, Clone)]
pub struct WasmMemoryStats {
    /// Number of allocations performed.
    pub allocation_count: u64,
    /// Number of deallocations performed.
    pub deallocation_count: u64,
    /// Memory fragmentation ratio.
    pub fragmentation_ratio: f32,
    /// Average allocation size.
    pub average_allocation_size: usize,
    /// Current memory pages in use.
    pub pages_used: usize,
}

impl WasmMemoryStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self {
            allocation_count: 0,
            deallocation_count: 0,
            fragmentation_ratio: 0.0,
            average_allocation_size: 0,
            pages_used: 0,
        }
    }

    /// Update statistics with a new allocation.
    pub fn record_allocation(&mut self, size: usize) {
        self.allocation_count += 1;

        // Update running average allocation size
        let total_count = self.allocation_count as usize;
        self.average_allocation_size =
            ((self.average_allocation_size * (total_count - 1)) + size) / total_count;
    }

    /// Update statistics with a deallocation.
    pub fn record_deallocation(&mut self) {
        self.deallocation_count += 1;
    }
}

impl Default for WasmMemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_allocator_creation() {
        let allocator = WasmMemoryAllocator::new();
        let memory_info = allocator.get_memory_info();

        assert_eq!(memory_info.allocated_bytes, 0);
        assert_eq!(memory_info.peak_bytes, 0);
        assert!(memory_info.total_bytes > 0);
    }

    #[test]
    fn test_memory_limit_enforcement() {
        let allocator = WasmMemoryAllocator::with_memory_limit(1024); // 1KB limit

        // This should fail as it exceeds the limit
        let result = allocator.allocate(&[1000], DataType::F32); // 4KB needed
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_allocation() -> Result<()> {
        let allocator = WasmMemoryAllocator::new();

        // Allocate a small F32 tensor
        let buffer = allocator.allocate(&[10, 10], DataType::F32)?;

        assert_eq!(buffer.size, 400); // 100 * 4 bytes
        assert_eq!(buffer.alignment, 16); // SIMD alignment
        assert_eq!(buffer.memory_type, MemoryType::SystemRAM);

        // Check memory stats
        let memory_info = allocator.get_memory_info();
        assert_eq!(memory_info.allocated_bytes, 400);

        // Deallocate
        allocator.deallocate(buffer)?;

        let memory_info_after = allocator.get_memory_info();
        assert_eq!(memory_info_after.allocated_bytes, 0);

        Ok(())
    }

    #[test]
    fn test_different_data_types() -> Result<()> {
        let allocator = WasmMemoryAllocator::new();

        // Test different data types
        let f32_buffer = allocator.allocate(&[100], DataType::F32)?;
        assert_eq!(f32_buffer.size, 400);

        let f16_buffer = allocator.allocate(&[100], DataType::F16)?;
        assert_eq!(f16_buffer.size, 200);

        let u8_buffer = allocator.allocate(&[100], DataType::U8)?;
        assert_eq!(u8_buffer.size, 100);

        // Clean up
        allocator.deallocate(f32_buffer)?;
        allocator.deallocate(f16_buffer)?;
        allocator.deallocate(u8_buffer)?;

        Ok(())
    }

    #[test]
    fn test_memory_utilization() {
        let allocator = WasmMemoryAllocator::with_memory_limit(1024);

        assert_eq!(allocator.get_memory_utilization(), 0.0);
        assert_eq!(allocator.get_available_pages(), 0); // 1024 < 64KB page size

        let large_allocator = WasmMemoryAllocator::with_memory_limit(128 * 1024); // 2 pages
        assert_eq!(large_allocator.get_available_pages(), 2);
    }

    #[test]
    fn test_wasm_memory_stats() {
        let mut stats = WasmMemoryStats::new();

        stats.record_allocation(1000);
        stats.record_allocation(2000);

        assert_eq!(stats.allocation_count, 2);
        assert_eq!(stats.average_allocation_size, 1500); // (1000 + 2000) / 2

        stats.record_deallocation();
        assert_eq!(stats.deallocation_count, 1);
    }
}
