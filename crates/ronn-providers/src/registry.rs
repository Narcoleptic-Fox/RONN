//! Provider registry and management system.
//!
//! The registry maintains a collection of execution providers and provides
//! capabilities for provider discovery, selection, and fallback chains.

use std::sync::{Arc, RwLock};

use anyhow::{Result, anyhow};
use dashmap::DashMap;
use ronn_core::{
    CompiledKernel, ExecutionProvider, OperatorSpec, ProviderCapability, ProviderId, SubGraph,
};
use tracing::{debug, info, warn};

/// Global provider registry managing all execution providers.
pub struct ProviderRegistry {
    /// Registered providers indexed by their ID.
    providers: DashMap<ProviderId, Arc<dyn ExecutionProvider>>,
    /// Provider preference order for fallback chains.
    preference_order: RwLock<Vec<ProviderId>>,
    /// Cached capability information for fast lookups.
    capabilities: DashMap<ProviderId, ProviderCapability>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    /// Create a new provider registry.
    pub fn new() -> Self {
        Self {
            providers: DashMap::new(),
            preference_order: RwLock::new(Vec::new()),
            capabilities: DashMap::new(),
        }
    }

    /// Register a new execution provider.
    ///
    /// # Arguments
    /// * `provider` - The execution provider to register
    ///
    /// # Returns
    /// * `Ok(())` - Provider registered successfully
    /// * `Err(anyhow::Error)` - Registration failed
    pub fn register_provider(&self, provider: Arc<dyn ExecutionProvider>) -> Result<()> {
        let provider_id = provider.provider_id();
        let capability = provider.get_capability();

        // Check for duplicate registration
        if self.providers.contains_key(&provider_id) {
            return Err(anyhow!("Provider {:?} is already registered", provider_id));
        }

        // Store provider and its capabilities
        self.providers.insert(provider_id, provider);
        self.capabilities.insert(provider_id, capability.clone());

        // Update preference order
        {
            let mut order = self.preference_order.write().unwrap();
            if !order.contains(&provider_id) {
                // Insert based on performance profile priority
                let insert_pos = match capability.performance_profile {
                    ronn_core::PerformanceProfile::GPU => 0, // Highest priority
                    ronn_core::PerformanceProfile::CPU => order.len().min(1),
                    ronn_core::PerformanceProfile::MemoryOptimized => order.len().min(2),
                    ronn_core::PerformanceProfile::PowerEfficient => order.len(), // Lowest priority
                };
                order.insert(insert_pos, provider_id);
            }
        }

        info!(
            "Registered provider {:?} with {} supported operations",
            provider_id,
            capability.supported_ops.len()
        );

        Ok(())
    }

    /// Unregister an execution provider.
    pub fn unregister_provider(&self, provider_id: ProviderId) -> Result<()> {
        let provider = self
            .providers
            .remove(&provider_id)
            .ok_or_else(|| anyhow!("Provider {:?} not found", provider_id))?;

        self.capabilities.remove(&provider_id);

        // Remove from preference order
        {
            let mut order = self.preference_order.write().unwrap();
            order.retain(|&id| id != provider_id);
        }

        // Shutdown the provider
        provider.1.shutdown()?;

        info!("Unregistered provider {:?}", provider_id);
        Ok(())
    }

    /// Get a provider by its ID.
    pub fn get_provider(&self, provider_id: ProviderId) -> Option<Arc<dyn ExecutionProvider>> {
        self.providers.get(&provider_id).map(|p| p.clone())
    }

    /// Get all registered provider IDs.
    pub fn get_provider_ids(&self) -> Vec<ProviderId> {
        self.providers.iter().map(|entry| *entry.key()).collect()
    }

    /// Get provider capabilities.
    pub fn get_capability(&self, provider_id: ProviderId) -> Option<ProviderCapability> {
        self.capabilities.get(&provider_id).map(|c| c.clone())
    }

    /// Select the best provider for a given set of operations.
    ///
    /// # Arguments
    /// * `operators` - The operations to execute
    ///
    /// # Returns
    /// * `Some(ProviderId)` - The best provider for these operations
    /// * `None` - No suitable provider found
    pub fn select_provider(&self, operators: &[OperatorSpec]) -> Option<ProviderId> {
        let order = self.preference_order.read().unwrap();

        for &provider_id in order.iter() {
            if let Some(provider) = self.providers.get(&provider_id) {
                let support_results = provider.can_handle(operators);

                // Check if provider can handle all operators
                if support_results.iter().all(|&supported| supported) {
                    debug!(
                        "Selected provider {:?} for {} operators",
                        provider_id,
                        operators.len()
                    );
                    return Some(provider_id);
                }
            }
        }

        warn!(
            "No provider found that can handle all {} operators",
            operators.len()
        );
        None
    }

    /// Get a fallback chain for operators that some providers can't handle.
    ///
    /// Returns a list of (provider_id, operator_indices) pairs indicating
    /// which provider should handle which operators.
    pub fn get_fallback_chain(&self, operators: &[OperatorSpec]) -> Vec<(ProviderId, Vec<usize>)> {
        let mut fallback_chain = Vec::new();
        let mut unhandled_ops: Vec<usize> = (0..operators.len()).collect();

        let order = self.preference_order.read().unwrap();

        for &provider_id in order.iter() {
            if unhandled_ops.is_empty() {
                break;
            }

            if let Some(provider) = self.providers.get(&provider_id) {
                let unhandled_operators: Vec<_> = unhandled_ops
                    .iter()
                    .map(|&i| operators[i].clone())
                    .collect();

                let support_results = provider.can_handle(&unhandled_operators);
                let mut handled_indices = Vec::new();

                for (local_idx, &supported) in support_results.iter().enumerate() {
                    if supported {
                        let original_idx = unhandled_ops[local_idx];
                        handled_indices.push(original_idx);
                    }
                }

                if !handled_indices.is_empty() {
                    fallback_chain.push((provider_id, handled_indices.clone()));

                    // Remove handled operators from unhandled list
                    unhandled_ops.retain(|idx| !handled_indices.contains(idx));
                }
            }
        }

        if !unhandled_ops.is_empty() {
            warn!(
                "Some operators could not be handled by any provider: {:?}",
                unhandled_ops
            );
        }

        debug!(
            "Generated fallback chain with {} providers for {} operators",
            fallback_chain.len(),
            operators.len()
        );

        fallback_chain
    }

    /// Set custom provider preference order.
    pub fn set_preference_order(&self, order: Vec<ProviderId>) -> Result<()> {
        // Validate that all providers in the order are registered
        for &provider_id in &order {
            if !self.providers.contains_key(&provider_id) {
                return Err(anyhow!("Provider {:?} is not registered", provider_id));
            }
        }

        {
            let mut preference = self.preference_order.write().unwrap();
            *preference = order;
        }

        info!("Updated provider preference order");
        Ok(())
    }

    /// Get the current provider preference order.
    pub fn get_preference_order(&self) -> Vec<ProviderId> {
        self.preference_order.read().unwrap().clone()
    }

    /// Compile a subgraph using the best available provider.
    pub fn compile_subgraph(
        &self,
        subgraph: SubGraph,
    ) -> Result<(ProviderId, Box<dyn CompiledKernel>)> {
        // Extract operator specs from subgraph
        let operators: Vec<OperatorSpec> = subgraph
            .nodes
            .iter()
            .map(|node| OperatorSpec {
                op_type: node.op_type.clone(),
                input_types: vec![],  // TODO: Extract from graph analysis
                output_types: vec![], // TODO: Extract from graph analysis
                attributes: node.attributes.clone(),
            })
            .collect();

        // Select the best provider
        let provider_id = self
            .select_provider(&operators)
            .ok_or_else(|| anyhow!("No provider available for subgraph compilation"))?;

        let provider = self
            .get_provider(provider_id)
            .ok_or_else(|| anyhow!("Provider {:?} not found", provider_id))?;

        // Compile the subgraph
        let kernel = provider.compile_subgraph(subgraph)?;

        Ok((provider_id, kernel))
    }

    /// Get registry statistics.
    pub fn get_statistics(&self) -> RegistryStatistics {
        let provider_count = self.providers.len();
        let total_supported_ops: usize = self
            .capabilities
            .iter()
            .map(|entry| entry.value().supported_ops.len())
            .sum();

        let preference_order = self.preference_order.read().unwrap().clone();

        RegistryStatistics {
            provider_count,
            total_supported_ops,
            preference_order,
        }
    }

    /// Shutdown all providers and cleanup.
    pub fn shutdown(&self) -> Result<()> {
        let provider_ids: Vec<_> = self.get_provider_ids();

        for provider_id in provider_ids {
            if let Err(e) = self.unregister_provider(provider_id) {
                warn!("Failed to shutdown provider {:?}: {}", provider_id, e);
            }
        }

        info!("Provider registry shutdown complete");
        Ok(())
    }
}

/// Statistics about the provider registry.
#[derive(Debug, Clone)]
pub struct RegistryStatistics {
    /// Number of registered providers.
    pub provider_count: usize,
    /// Total number of supported operations across all providers.
    pub total_supported_ops: usize,
    /// Current provider preference order.
    pub preference_order: Vec<ProviderId>,
}

// Thread-safety: ProviderRegistry uses DashMap and RwLock for thread-safe access
// These are automatically implemented since DashMap and RwLock are Send + Sync

#[cfg(test)]
mod tests {
    use super::*;
    use ronn_core::{
        AttributeValue, DataType, KernelStats, MemoryType, MemoryUsage, PerformanceProfile,
        ProviderConfig, ResourceRequirements, Tensor, TensorAllocator,
    };
    use std::collections::{HashMap, HashSet};

    // Mock execution provider for testing
    struct MockProvider {
        id: ProviderId,
        supported_ops: HashSet<String>,
        performance_profile: PerformanceProfile,
    }

    impl MockProvider {
        fn new(id: ProviderId, ops: Vec<&str>, profile: PerformanceProfile) -> Self {
            Self {
                id,
                supported_ops: ops.into_iter().map(|s| s.to_string()).collect(),
                performance_profile: profile,
            }
        }
    }

    impl ExecutionProvider for MockProvider {
        fn provider_id(&self) -> ProviderId {
            self.id
        }

        fn get_capability(&self) -> ProviderCapability {
            ProviderCapability {
                supported_ops: self.supported_ops.clone(),
                data_types: vec![DataType::F32, DataType::F16],
                memory_types: vec![MemoryType::SystemRAM],
                performance_profile: self.performance_profile,
                resource_requirements: ResourceRequirements {
                    min_memory_bytes: Some(1024 * 1024), // 1MB
                    cpu_features: vec![],
                    gpu_memory_bytes: None,
                },
            }
        }

        fn can_handle(&self, operators: &[OperatorSpec]) -> Vec<bool> {
            operators
                .iter()
                .map(|op| self.supported_ops.contains(&op.op_type))
                .collect()
        }

        fn compile_subgraph(&self, _subgraph: SubGraph) -> anyhow::Result<Box<dyn CompiledKernel>> {
            Ok(Box::new(MockKernel))
        }

        fn get_allocator(&self) -> std::sync::Arc<dyn TensorAllocator> {
            unimplemented!("Mock provider doesn't implement allocator")
        }

        fn configure(&mut self, _config: ProviderConfig) -> anyhow::Result<()> {
            Ok(())
        }

        fn shutdown(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct MockKernel;

    impl CompiledKernel for MockKernel {
        fn execute(&self, _inputs: &[Tensor]) -> anyhow::Result<Vec<Tensor>> {
            Ok(vec![])
        }

        fn get_memory_usage(&self) -> MemoryUsage {
            MemoryUsage {
                peak_bytes: 1024,
                current_bytes: 512,
                allocation_count: 1,
            }
        }

        fn get_performance_stats(&self) -> KernelStats {
            KernelStats {
                execution_count: 1,
                average_time_us: 100.0,
                min_time_us: 90.0,
                max_time_us: 110.0,
            }
        }
    }

    #[test]
    fn test_provider_registration() -> Result<()> {
        let registry = ProviderRegistry::new();
        let provider = Arc::new(MockProvider::new(
            ProviderId::CPU,
            vec!["Add", "Mul"],
            PerformanceProfile::CPU,
        ));

        registry.register_provider(provider)?;

        assert_eq!(registry.get_provider_ids().len(), 1);
        assert!(registry.get_provider(ProviderId::CPU).is_some());

        Ok(())
    }

    #[test]
    fn test_provider_selection() -> Result<()> {
        let registry = ProviderRegistry::new();

        let cpu_provider = Arc::new(MockProvider::new(
            ProviderId::CPU,
            vec!["Add", "Mul"],
            PerformanceProfile::CPU,
        ));

        let gpu_provider = Arc::new(MockProvider::new(
            ProviderId::GPU,
            vec!["Conv", "MatMul"],
            PerformanceProfile::GPU,
        ));

        registry.register_provider(cpu_provider)?;
        registry.register_provider(gpu_provider)?;

        // Test operator selection
        let add_op = OperatorSpec {
            op_type: "Add".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        };

        let conv_op = OperatorSpec {
            op_type: "Conv".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        };

        // CPU provider should be selected for Add operation
        let selected = registry.select_provider(&[add_op]);
        assert_eq!(selected, Some(ProviderId::CPU));

        // GPU provider should be selected for Conv operation (higher priority)
        let selected = registry.select_provider(&[conv_op]);
        assert_eq!(selected, Some(ProviderId::GPU));

        Ok(())
    }

    #[test]
    fn test_fallback_chain() -> Result<()> {
        let registry = ProviderRegistry::new();

        let cpu_provider = Arc::new(MockProvider::new(
            ProviderId::CPU,
            vec!["Add", "Mul"],
            PerformanceProfile::CPU,
        ));

        let gpu_provider = Arc::new(MockProvider::new(
            ProviderId::GPU,
            vec!["Conv"],
            PerformanceProfile::GPU,
        ));

        registry.register_provider(cpu_provider)?;
        registry.register_provider(gpu_provider)?;

        let operators = vec![
            OperatorSpec {
                op_type: "Add".to_string(),
                input_types: vec![DataType::F32],
                output_types: vec![DataType::F32],
                attributes: HashMap::new(),
            },
            OperatorSpec {
                op_type: "Conv".to_string(),
                input_types: vec![DataType::F32],
                output_types: vec![DataType::F32],
                attributes: HashMap::new(),
            },
        ];

        let fallback_chain = registry.get_fallback_chain(&operators);

        // Should have GPU provider handling Conv (index 1) and CPU handling Add (index 0)
        assert_eq!(fallback_chain.len(), 2);

        // GPU provider should handle Conv operation (index 1)
        let gpu_entry = fallback_chain
            .iter()
            .find(|(id, _)| *id == ProviderId::GPU)
            .unwrap();
        assert_eq!(gpu_entry.1, vec![1]);

        // CPU provider should handle Add operation (index 0)
        let cpu_entry = fallback_chain
            .iter()
            .find(|(id, _)| *id == ProviderId::CPU)
            .unwrap();
        assert_eq!(cpu_entry.1, vec![0]);

        Ok(())
    }

    #[test]
    fn test_preference_order() -> Result<()> {
        let registry = ProviderRegistry::new();

        let cpu_provider = Arc::new(MockProvider::new(
            ProviderId::CPU,
            vec!["Add"],
            PerformanceProfile::CPU,
        ));

        let gpu_provider = Arc::new(MockProvider::new(
            ProviderId::GPU,
            vec!["Add"],
            PerformanceProfile::GPU,
        ));

        registry.register_provider(cpu_provider)?;
        registry.register_provider(gpu_provider)?;

        // Default order should prioritize GPU
        let order = registry.get_preference_order();
        assert_eq!(order[0], ProviderId::GPU);

        // Change preference to CPU first
        registry.set_preference_order(vec![ProviderId::CPU, ProviderId::GPU])?;

        let operators = vec![OperatorSpec {
            op_type: "Add".to_string(),
            input_types: vec![DataType::F32],
            output_types: vec![DataType::F32],
            attributes: HashMap::new(),
        }];

        // Now CPU should be selected first
        let selected = registry.select_provider(&operators);
        assert_eq!(selected, Some(ProviderId::CPU));

        Ok(())
    }
}
