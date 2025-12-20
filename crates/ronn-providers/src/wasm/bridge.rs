//! WebAssembly bridge for JavaScript interoperability.
//!
//! This module provides seamless integration between Rust/WASM and JavaScript,
//! including TypedArray conversion, IndexedDB caching, and Web Worker support.

use anyhow::{Result, anyhow};
use ronn_core::{DataType, Tensor, TensorLayout};
use std::collections::HashMap;

/// Bridge for JavaScript interoperability.
#[derive(Debug)]
pub struct WasmBridge {
    /// Cached model data.
    cache: IndexedDbCache,
    /// JavaScript interface for TypedArrays.
    typed_array_interface: TypedArrayInterface,
    /// Configuration options.
    config: WasmBridgeConfig,
}

/// Configuration for the WASM bridge.
#[derive(Debug, Clone)]
pub struct WasmBridgeConfig {
    /// Enable IndexedDB caching.
    pub enable_caching: bool,
    /// Maximum cache size in bytes.
    pub max_cache_size: usize,
    /// Enable Web Worker support.
    pub enable_web_workers: bool,
    /// Number of worker threads.
    pub worker_count: usize,
    /// Cache expiration time in milliseconds.
    pub cache_expiry_ms: u64,
}

impl Default for WasmBridgeConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 128 * 1024 * 1024, // 128MB
            enable_web_workers: true,
            worker_count: navigator_hardware_concurrency().max(1),
            cache_expiry_ms: 24 * 60 * 60 * 1000, // 24 hours
        }
    }
}

/// Get the number of logical processors (fallback implementation).
fn navigator_hardware_concurrency() -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        // In a real WASM environment, this would query navigator.hardwareConcurrency
        4 // Default fallback
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        num_cpus::get()
    }
}

/// JavaScript TypedArray interface.
#[derive(Debug, Clone)]
pub struct TypedArrayInterface;

impl TypedArrayInterface {
    /// Convert Rust tensor to JavaScript TypedArray.
    pub fn tensor_to_typed_array(&self, tensor: &Tensor) -> Result<TypedArrayData> {
        let data = tensor.to_vec()?;
        match tensor.dtype() {
            DataType::F32 => Ok(TypedArrayData::Float32(data)),
            DataType::F16 => {
                // Convert F16 data to F32 for JavaScript compatibility
                // Simplified - would need proper F16 conversion
                Ok(TypedArrayData::Float32(data))
            }
            DataType::U8 => {
                let u8_data: Vec<u8> = data.iter().map(|&x| x as u8).collect();
                Ok(TypedArrayData::Uint8(u8_data))
            }
            DataType::I8 => {
                let i8_data: Vec<i8> = data.iter().map(|&x| x as i8).collect();
                Ok(TypedArrayData::Int8(i8_data))
            }
            DataType::I32 => {
                let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                Ok(TypedArrayData::Int32(i32_data))
            }
            DataType::U32 => {
                let u32_data: Vec<u32> = data.iter().map(|&x| x as u32).collect();
                Ok(TypedArrayData::Uint32(u32_data))
            }
            DataType::Bool => {
                let u8_data: Vec<u8> = data.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).collect();
                Ok(TypedArrayData::Uint8(u8_data))
            }
            _ => Err(anyhow!(
                "Unsupported data type for TypedArray conversion: {:?}",
                tensor.dtype()
            )),
        }
    }

    /// Convert JavaScript TypedArray to Rust tensor.
    pub fn typed_array_to_tensor(
        &self,
        data: TypedArrayData,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> Result<Tensor> {
        let f32_data = match data {
            TypedArrayData::Float32(data) => data,
            TypedArrayData::Float64(data) => data.iter().map(|&x| x as f32).collect(),
            TypedArrayData::Uint8(data) => data.iter().map(|&x| x as f32).collect(),
            TypedArrayData::Int8(data) => data.iter().map(|&x| x as f32).collect(),
            TypedArrayData::Uint32(data) => data.iter().map(|&x| x as f32).collect(),
            TypedArrayData::Int32(data) => data.iter().map(|&x| x as f32).collect(),
        };

        Tensor::from_data(f32_data, shape, dtype, TensorLayout::RowMajor)
    }

    /// Get optimal batch size for the current browser.
    pub fn get_optimal_batch_size(&self, tensor_size: usize) -> usize {
        // Heuristic based on available memory and tensor size
        let available_memory = self.estimate_available_memory();
        let memory_per_tensor = tensor_size * std::mem::size_of::<f32>();

        if memory_per_tensor == 0 {
            return 1;
        }

        // Use at most 25% of available memory for batching
        let max_memory_for_batch = available_memory / 4;
        (max_memory_for_batch / memory_per_tensor).max(1).min(64) // Cap at 64
    }

    /// Estimate available browser memory.
    fn estimate_available_memory(&self) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            // In a real implementation, this would use performance.memory or similar
            512 * 1024 * 1024 // 512MB estimate
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            1024 * 1024 * 1024 // 1GB for testing
        }
    }
}

/// TypedArray data variants for JavaScript interop.
#[derive(Debug, Clone)]
pub enum TypedArrayData {
    /// Float32Array
    Float32(Vec<f32>),
    /// Float64Array
    Float64(Vec<f64>),
    /// Uint8Array
    Uint8(Vec<u8>),
    /// Int8Array
    Int8(Vec<i8>),
    /// Uint32Array
    Uint32(Vec<u32>),
    /// Int32Array
    Int32(Vec<i32>),
}

/// IndexedDB cache for model and tensor data.
#[derive(Debug)]
pub struct IndexedDbCache {
    /// In-memory cache (simplified - real implementation would use IndexedDB).
    memory_cache: HashMap<String, CacheEntry>,
    /// Maximum cache size.
    max_size: usize,
    /// Current cache size.
    current_size: usize,
}

/// Cache entry with metadata.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached data.
    data: Vec<u8>,
    /// Timestamp when cached.
    timestamp: u64,
    /// Access count for LRU.
    access_count: u64,
    /// Data size in bytes.
    size: usize,
}

impl IndexedDbCache {
    /// Create a new IndexedDB cache.
    pub fn new(max_size: usize) -> Self {
        Self {
            memory_cache: HashMap::new(),
            max_size,
            current_size: 0,
        }
    }

    /// Store data in the cache.
    pub async fn store(&mut self, key: &str, data: &[u8]) -> Result<()> {
        let entry = CacheEntry {
            data: data.to_vec(),
            timestamp: current_timestamp_ms(),
            access_count: 0,
            size: data.len(),
        };

        // Check if we need to evict entries
        while self.current_size + entry.size > self.max_size && !self.memory_cache.is_empty() {
            self.evict_lru_entry();
        }

        if entry.size <= self.max_size {
            self.current_size += entry.size;
            self.memory_cache.insert(key.to_string(), entry);
        }

        Ok(())
    }

    /// Retrieve data from the cache.
    pub async fn retrieve(&mut self, key: &str) -> Option<Vec<u8>> {
        if let Some(entry) = self.memory_cache.get_mut(key) {
            // Check if entry has expired
            let current_time = current_timestamp_ms();
            if current_time - entry.timestamp > 24 * 60 * 60 * 1000 {
                // 24 hours
                return None;
            }

            entry.access_count += 1;
            Some(entry.data.clone())
        } else {
            None
        }
    }

    /// Clear the entire cache.
    pub async fn clear(&mut self) -> Result<()> {
        self.memory_cache.clear();
        self.current_size = 0;
        Ok(())
    }

    /// Get cache statistics.
    pub fn get_stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.memory_cache.len(),
            total_size: self.current_size,
            max_size: self.max_size,
            hit_rate: 0.0, // Would need to track hits/misses
        }
    }

    /// Evict the least recently used entry.
    fn evict_lru_entry(&mut self) {
        let lru_key = self
            .memory_cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            if let Some(entry) = self.memory_cache.remove(&key) {
                self.current_size -= entry.size;
            }
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache entries.
    pub entry_count: usize,
    /// Total cache size in bytes.
    pub total_size: usize,
    /// Maximum cache size.
    pub max_size: usize,
    /// Cache hit rate (0.0 to 1.0).
    pub hit_rate: f32,
}

/// Get current timestamp in milliseconds.
fn current_timestamp_ms() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        // In real WASM, this would use Date.now() or performance.now()
        0 // Simplified for testing
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
}

impl WasmBridge {
    /// Create a new WASM bridge with default configuration.
    pub fn new() -> Self {
        Self::with_config(WasmBridgeConfig::default())
    }

    /// Create a WASM bridge with custom configuration.
    pub fn with_config(config: WasmBridgeConfig) -> Self {
        let cache = IndexedDbCache::new(config.max_cache_size);
        let typed_array_interface = TypedArrayInterface;

        Self {
            cache,
            typed_array_interface,
            config,
        }
    }

    /// Convert a tensor to JavaScript-compatible format.
    pub fn export_tensor(&self, tensor: &Tensor) -> Result<TensorExport> {
        let typed_array = self.typed_array_interface.tensor_to_typed_array(tensor)?;

        Ok(TensorExport {
            data: typed_array,
            shape: tensor.shape().to_vec(),
            dtype: format!("{:?}", tensor.dtype()),
        })
    }

    /// Import a tensor from JavaScript format.
    pub fn import_tensor(&self, export: TensorImport) -> Result<Tensor> {
        let dtype = match export.dtype.as_str() {
            "F32" => DataType::F32,
            "F16" => DataType::F16,
            "U8" => DataType::U8,
            "I8" => DataType::I8,
            "I32" => DataType::I32,
            "U32" => DataType::U32,
            "Bool" => DataType::Bool,
            _ => return Err(anyhow!("Unknown data type: {}", export.dtype)),
        };

        self.typed_array_interface
            .typed_array_to_tensor(export.data, export.shape, dtype)
    }

    /// Cache model data for future use.
    pub async fn cache_model_data(&mut self, model_id: &str, data: &[u8]) -> Result<()> {
        if self.config.enable_caching {
            self.cache.store(model_id, data).await?;
        }
        Ok(())
    }

    /// Retrieve cached model data.
    pub async fn get_cached_model_data(&mut self, model_id: &str) -> Option<Vec<u8>> {
        if self.config.enable_caching {
            self.cache.retrieve(model_id).await
        } else {
            None
        }
    }

    /// Get cache statistics.
    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache.get_stats()
    }

    /// Initialize Web Workers for parallel processing.
    pub async fn initialize_workers(&self) -> Result<WorkerPool> {
        if !self.config.enable_web_workers {
            return Ok(WorkerPool::new(0));
        }

        Ok(WorkerPool::new(self.config.worker_count))
    }
}

impl Default for WasmBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor data for export to JavaScript.
#[derive(Debug, Clone)]
pub struct TensorExport {
    /// TypedArray data.
    pub data: TypedArrayData,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Data type name.
    pub dtype: String,
}

/// Tensor data imported from JavaScript.
#[derive(Debug, Clone)]
pub struct TensorImport {
    /// TypedArray data.
    pub data: TypedArrayData,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Data type name.
    pub dtype: String,
}

/// Web Worker pool for parallel processing.
#[derive(Debug)]
pub struct WorkerPool {
    /// Number of workers.
    worker_count: usize,
    /// Worker availability.
    available_workers: Vec<bool>,
}

impl WorkerPool {
    /// Create a new worker pool.
    pub fn new(worker_count: usize) -> Self {
        Self {
            worker_count,
            available_workers: vec![true; worker_count],
        }
    }

    /// Get the number of available workers.
    pub fn available_count(&self) -> usize {
        self.available_workers
            .iter()
            .filter(|&&available| available)
            .count()
    }

    /// Reserve a worker for processing.
    pub fn reserve_worker(&mut self) -> Option<usize> {
        for (i, available) in self.available_workers.iter_mut().enumerate() {
            if *available {
                *available = false;
                return Some(i);
            }
        }
        None
    }

    /// Release a worker back to the pool.
    pub fn release_worker(&mut self, worker_id: usize) {
        if worker_id < self.worker_count {
            self.available_workers[worker_id] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_bridge_creation() {
        let bridge = WasmBridge::new();
        assert!(bridge.config.enable_caching);
        assert!(bridge.config.enable_web_workers);
    }

    #[test]
    fn test_tensor_export_import() -> Result<()> {
        let bridge = WasmBridge::new();

        // Create a test tensor
        let original = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        // Export to JavaScript format
        let exported = bridge.export_tensor(&original)?;

        // Import back to tensor
        let imported_data = TensorImport {
            data: exported.data,
            shape: exported.shape,
            dtype: exported.dtype,
        };
        let imported = bridge.import_tensor(imported_data)?;

        // Verify roundtrip
        assert_eq!(original.shape(), imported.shape());
        assert_eq!(original.to_vec().unwrap(), imported.to_vec().unwrap());

        Ok(())
    }

    #[test]
    fn test_typed_array_interface() -> Result<()> {
        let interface = TypedArrayInterface;

        let tensor = Tensor::from_data(
            vec![1.0, -2.0, 3.5],
            vec![3],
            DataType::F32,
            TensorLayout::RowMajor,
        )?;

        let typed_array = interface.tensor_to_typed_array(&tensor)?;

        match typed_array {
            TypedArrayData::Float32(data) => {
                assert_eq!(data, vec![1.0, -2.0, 3.5]);
            }
            _ => panic!("Expected Float32 array"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_indexeddb_cache() -> Result<()> {
        let mut cache = IndexedDbCache::new(1024); // 1KB cache

        let test_data = vec![1, 2, 3, 4, 5];

        // Store data
        cache.store("test_key", &test_data).await?;

        // Retrieve data
        let retrieved = cache.retrieve("test_key").await;
        assert_eq!(retrieved, Some(test_data));

        // Test cache statistics
        let stats = cache.get_stats();
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.total_size, 5);

        Ok(())
    }

    #[test]
    fn test_optimal_batch_size() {
        let interface = TypedArrayInterface;

        let batch_size = interface.get_optimal_batch_size(1000);
        assert!(batch_size > 0);
        assert!(batch_size <= 64);
    }

    #[tokio::test]
    async fn test_worker_pool() {
        let mut pool = WorkerPool::new(4);

        assert_eq!(pool.available_count(), 4);

        let worker1 = pool.reserve_worker();
        assert_eq!(worker1, Some(0));
        assert_eq!(pool.available_count(), 3);

        let worker2 = pool.reserve_worker();
        assert_eq!(worker2, Some(1));
        assert_eq!(pool.available_count(), 2);

        pool.release_worker(0);
        assert_eq!(pool.available_count(), 3);
    }
}
