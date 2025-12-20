//! Dynamic provider registry for custom hardware plugins.
//!
//! This module implements a plugin-based architecture for loading and managing
//! custom hardware providers at runtime.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

use super::traits::{CustomHardwareProvider, HardwareDevice, HardwareDiscovery};

/// Registry for managing custom hardware provider plugins.
#[derive(Debug)]
pub struct CustomProviderRegistry {
    /// Registered providers by name.
    providers: Arc<RwLock<HashMap<String, Arc<dyn CustomHardwareProvider>>>>,
    /// Plugin metadata.
    plugin_metadata: HashMap<String, PluginMetadata>,
    /// Plugin search paths.
    plugin_paths: Vec<PathBuf>,
    /// Hardware discovery service.
    hardware_discovery: Option<Box<dyn HardwareDiscovery>>,
}

/// Metadata for a loaded plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin name.
    pub name: String,
    /// Plugin version.
    pub version: String,
    /// Plugin author/vendor.
    pub author: String,
    /// Plugin description.
    pub description: String,
    /// Plugin license.
    pub license: String,
    /// Minimum RONN version required.
    pub min_ronn_version: String,
    /// Supported hardware types.
    pub supported_hardware: Vec<String>,
    /// Plugin ABI version.
    pub abi_version: u32,
    /// Plugin file path.
    pub plugin_path: PathBuf,
    /// Load status.
    pub status: PluginStatus,
}

/// Status of a plugin.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PluginStatus {
    /// Plugin is loaded and active.
    Loaded,
    /// Plugin failed to load.
    LoadError(String),
    /// Plugin is disabled.
    Disabled,
    /// Plugin version is incompatible.
    Incompatible,
}

/// Plugin interface for dynamic loading.
pub trait ProviderPlugin: Send + Sync {
    /// Get plugin metadata.
    fn get_metadata(&self) -> PluginMetadata;

    /// Create a provider instance.
    fn create_provider(&self, config: &str) -> Result<Box<dyn CustomHardwareProvider>>;

    /// Check if the required hardware is available.
    fn is_hardware_available(&self) -> bool;

    /// Get plugin ABI version.
    fn get_abi_version(&self) -> u32 {
        1 // Current ABI version
    }
}

impl CustomProviderRegistry {
    /// Create a new custom provider registry.
    pub fn new() -> Self {
        let default_paths = vec![
            PathBuf::from("./plugins"),
            PathBuf::from("/usr/local/lib/ronn/plugins"),
            PathBuf::from("/opt/ronn/plugins"),
        ];

        Self {
            providers: Arc::new(RwLock::new(HashMap::new())),
            plugin_metadata: HashMap::new(),
            plugin_paths: default_paths,
            hardware_discovery: None,
        }
    }

    /// Add a plugin search path.
    pub fn add_plugin_path<P: AsRef<Path>>(&mut self, path: P) {
        let path = path.as_ref().to_path_buf();
        if !self.plugin_paths.contains(&path) {
            self.plugin_paths.push(path);
        }
    }

    /// Set hardware discovery service.
    pub fn set_hardware_discovery(&mut self, discovery: Box<dyn HardwareDiscovery>) {
        self.hardware_discovery = Some(discovery);
    }

    /// Discover and load plugins from search paths.
    pub fn discover_plugins(&mut self) -> Result<Vec<PluginMetadata>> {
        let mut discovered_plugins = Vec::new();

        for plugin_path in &self.plugin_paths.clone() {
            if plugin_path.exists() {
                match self.scan_plugin_directory(plugin_path) {
                    Ok(mut plugins) => {
                        discovered_plugins.append(&mut plugins);
                    }
                    Err(e) => {
                        warn!("Failed to scan plugin directory {:?}: {}", plugin_path, e);
                    }
                }
            }
        }

        info!("Discovered {} plugins", discovered_plugins.len());
        Ok(discovered_plugins)
    }

    /// Scan a directory for plugins.
    fn scan_plugin_directory(&mut self, dir: &Path) -> Result<Vec<PluginMetadata>> {
        let mut plugins = Vec::new();

        if !dir.is_dir() {
            return Ok(plugins);
        }

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            // Look for dynamic libraries
            if let Some(extension) = path.extension() {
                let is_plugin = match extension.to_str() {
                    Some("so") => true,    // Linux
                    Some("dylib") => true, // macOS
                    Some("dll") => true,   // Windows
                    _ => false,
                };

                if is_plugin {
                    match self.load_plugin_metadata(&path) {
                        Ok(metadata) => {
                            plugins.push(metadata);
                        }
                        Err(e) => {
                            error!("Failed to load plugin metadata from {:?}: {}", path, e);
                        }
                    }
                }
            }
        }

        Ok(plugins)
    }

    /// Load plugin metadata from a plugin file.
    fn load_plugin_metadata(&mut self, plugin_path: &Path) -> Result<PluginMetadata> {
        // In a real implementation, this would use dynamic library loading
        // For now, we'll create a mock metadata entry

        let plugin_name = plugin_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let metadata = PluginMetadata {
            name: plugin_name.clone(),
            version: "1.0.0".to_string(),
            author: "Unknown".to_string(),
            description: format!("Custom hardware provider plugin: {}", plugin_name),
            license: "MIT".to_string(),
            min_ronn_version: "0.1.0".to_string(),
            supported_hardware: vec!["Custom".to_string()],
            abi_version: 1,
            plugin_path: plugin_path.to_path_buf(),
            status: PluginStatus::Loaded,
        };

        self.plugin_metadata.insert(plugin_name, metadata.clone());
        Ok(metadata)
    }

    /// Register a provider directly (for statically linked providers).
    pub fn register_provider(
        &mut self,
        name: String,
        provider: Arc<dyn CustomHardwareProvider>,
    ) -> Result<()> {
        let mut providers = self
            .providers
            .write()
            .map_err(|_| anyhow!("Lock poisoned"))?;

        if providers.contains_key(&name) {
            return Err(anyhow!("Provider {} already registered", name));
        }

        providers.insert(name.clone(), provider);
        info!("Registered custom hardware provider: {}", name);
        Ok(())
    }

    /// Get a provider by name.
    pub fn get_provider(&self, name: &str) -> Option<Arc<dyn CustomHardwareProvider>> {
        let providers = self.providers.read().ok()?;
        providers.get(name).cloned()
    }

    /// List all registered providers.
    pub fn list_providers(&self) -> Vec<String> {
        self.providers
            .read()
            .map(|providers| providers.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get provider metadata.
    pub fn get_plugin_metadata(&self, name: &str) -> Option<&PluginMetadata> {
        self.plugin_metadata.get(name)
    }

    /// Get all plugin metadata.
    pub fn get_all_plugin_metadata(&self) -> &HashMap<String, PluginMetadata> {
        &self.plugin_metadata
    }

    /// Unregister a provider.
    pub fn unregister_provider(&mut self, name: &str) -> Result<()> {
        let mut providers = self
            .providers
            .write()
            .map_err(|_| anyhow!("Lock poisoned"))?;

        if providers.remove(name).is_some() {
            self.plugin_metadata.remove(name);
            info!("Unregistered provider: {}", name);
            Ok(())
        } else {
            Err(anyhow!("Provider {} not found", name))
        }
    }

    /// Discover available hardware devices.
    pub fn discover_hardware(&self) -> Result<Vec<HardwareDevice>> {
        if let Some(ref discovery) = self.hardware_discovery {
            discovery.discover_devices()
        } else {
            warn!("No hardware discovery service configured");
            Ok(Vec::new())
        }
    }

    /// Get registry statistics.
    pub fn get_statistics(&self) -> RegistryStatistics {
        let provider_count = self
            .providers
            .read()
            .map(|providers| providers.len())
            .unwrap_or(0);

        let plugin_status_counts =
            self.plugin_metadata
                .values()
                .fold(HashMap::new(), |mut acc, metadata| {
                    let status_key = match &metadata.status {
                        PluginStatus::Loaded => "loaded",
                        PluginStatus::LoadError(_) => "error",
                        PluginStatus::Disabled => "disabled",
                        PluginStatus::Incompatible => "incompatible",
                    };
                    *acc.entry(status_key.to_string()).or_insert(0) += 1;
                    acc
                });

        RegistryStatistics {
            registered_providers: provider_count,
            discovered_plugins: self.plugin_metadata.len(),
            plugin_paths: self.plugin_paths.clone(),
            plugin_status_counts,
            has_hardware_discovery: self.hardware_discovery.is_some(),
        }
    }

    /// Load a plugin from a specific file.
    pub fn load_plugin_from_file<P: AsRef<Path>>(&mut self, plugin_path: P) -> Result<()> {
        let plugin_path = plugin_path.as_ref();

        if !plugin_path.exists() {
            return Err(anyhow!("Plugin file does not exist: {:?}", plugin_path));
        }

        // Load metadata
        let metadata = self.load_plugin_metadata(plugin_path)?;

        // In a real implementation, this would:
        // 1. Load the dynamic library using dlopen/LoadLibrary
        // 2. Get the plugin entry point function
        // 3. Call the entry point to get the plugin instance
        // 4. Create and register the provider

        info!("Loaded plugin: {} v{}", metadata.name, metadata.version);
        Ok(())
    }

    /// Validate plugin compatibility.
    fn validate_plugin(&self, metadata: &PluginMetadata) -> Result<()> {
        // Check ABI version
        const CURRENT_ABI_VERSION: u32 = 1;
        if metadata.abi_version != CURRENT_ABI_VERSION {
            return Err(anyhow!(
                "Plugin {} has incompatible ABI version: {} (expected {})",
                metadata.name,
                metadata.abi_version,
                CURRENT_ABI_VERSION
            ));
        }

        // Check minimum RONN version
        // In a real implementation, this would do proper version comparison
        if metadata.min_ronn_version.is_empty() {
            warn!(
                "Plugin {} does not specify minimum RONN version",
                metadata.name
            );
        }

        Ok(())
    }

    /// Cleanup and shutdown all providers.
    pub fn shutdown(&mut self) -> Result<()> {
        let providers = self
            .providers
            .read()
            .map_err(|_| anyhow!("Lock poisoned"))?;

        for (name, _provider) in providers.iter() {
            debug!("Shutting down provider: {}", name);
            // In a real implementation, would call provider.shutdown()
        }

        info!("Custom provider registry shutdown complete");
        Ok(())
    }
}

impl Default for CustomProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the provider registry.
#[derive(Debug, Clone)]
pub struct RegistryStatistics {
    /// Number of registered providers.
    pub registered_providers: usize,
    /// Number of discovered plugins.
    pub discovered_plugins: usize,
    /// Plugin search paths.
    pub plugin_paths: Vec<PathBuf>,
    /// Count of plugins by status.
    pub plugin_status_counts: HashMap<String, usize>,
    /// Whether hardware discovery is available.
    pub has_hardware_discovery: bool,
}

/// Simple hardware discovery implementation.
#[derive(Debug)]
pub struct DefaultHardwareDiscovery;

impl HardwareDiscovery for DefaultHardwareDiscovery {
    fn discover_devices(&self) -> Result<Vec<HardwareDevice>> {
        // In a real implementation, this would scan for actual hardware
        // For now, return empty list
        Ok(Vec::new())
    }

    fn is_device_available(&self, _device_id: &str) -> bool {
        false
    }

    fn get_device_info(&self, _device_id: &str) -> Option<HardwareDevice> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // Mock provider for testing
    #[derive(Debug)]
    struct MockProvider {
        name: String,
    }

    impl CustomHardwareProvider for MockProvider {
        fn provider_name(&self) -> &str {
            &self.name
        }

        fn get_hardware_capability(&self) -> super::super::traits::HardwareCapability {
            super::super::traits::HardwareCapability {
                vendor: "Mock".to_string(),
                model: "TestDevice".to_string(),
                architecture_version: "1.0".to_string(),
                supported_data_types: vec![ronn_core::DataType::F32],
                max_memory_bytes: 1024 * 1024 * 1024,
                peak_tops: 10.0,
                memory_bandwidth_gbps: 100.0,
                supported_operations: vec!["Add".to_string()],
                features: HashMap::new(),
                power_profile: super::super::traits::PowerProfile {
                    idle_power_watts: 1.0,
                    peak_power_watts: 10.0,
                    tdp_watts: 5.0,
                    efficiency_tops_per_watt: 2.0,
                },
            }
        }

        fn is_hardware_available(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn compile_subgraph(
            &self,
            _subgraph: &ronn_core::SubGraph,
        ) -> Result<Box<dyn super::super::traits::CustomKernel>> {
            Err(anyhow!("Not implemented"))
        }

        fn get_device_memory(&self) -> &dyn super::super::traits::DeviceMemory {
            panic!("Not implemented")
        }

        fn get_performance_stats(&self) -> super::super::traits::ProviderStats {
            super::super::traits::ProviderStats {
                total_operations: 0,
                average_execution_time_us: 0.0,
                memory_usage_bytes: 0,
                peak_memory_bytes: 0,
                hardware_utilization: 0.0,
                current_power_watts: 0.0,
                total_energy_joules: 0.0,
            }
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    #[test]
    fn test_registry_creation() {
        let registry = CustomProviderRegistry::new();
        assert!(registry.list_providers().is_empty());
    }

    #[test]
    fn test_provider_registration() -> Result<()> {
        let mut registry = CustomProviderRegistry::new();
        let provider = Arc::new(MockProvider {
            name: "test_provider".to_string(),
        });

        registry.register_provider("test_provider".to_string(), provider.clone())?;

        let providers = registry.list_providers();
        assert_eq!(providers.len(), 1);
        assert!(providers.contains(&"test_provider".to_string()));

        let retrieved_provider = registry.get_provider("test_provider");
        assert!(retrieved_provider.is_some());

        Ok(())
    }

    #[test]
    fn test_duplicate_registration() {
        let mut registry = CustomProviderRegistry::new();
        let provider1 = Arc::new(MockProvider {
            name: "test_provider".to_string(),
        });
        let provider2 = Arc::new(MockProvider {
            name: "test_provider".to_string(),
        });

        registry
            .register_provider("test_provider".to_string(), provider1)
            .unwrap();

        let result = registry.register_provider("test_provider".to_string(), provider2);
        assert!(result.is_err());
    }

    #[test]
    fn test_provider_unregistration() -> Result<()> {
        let mut registry = CustomProviderRegistry::new();
        let provider = Arc::new(MockProvider {
            name: "test_provider".to_string(),
        });

        registry.register_provider("test_provider".to_string(), provider)?;
        assert_eq!(registry.list_providers().len(), 1);

        registry.unregister_provider("test_provider")?;
        assert_eq!(registry.list_providers().len(), 0);

        Ok(())
    }

    #[test]
    fn test_registry_statistics() -> Result<()> {
        let mut registry = CustomProviderRegistry::new();
        let provider = Arc::new(MockProvider {
            name: "test_provider".to_string(),
        });

        registry.register_provider("test_provider".to_string(), provider)?;

        let stats = registry.get_statistics();
        assert_eq!(stats.registered_providers, 1);
        assert!(!stats.plugin_paths.is_empty());
        assert!(!stats.has_hardware_discovery);

        Ok(())
    }

    #[test]
    fn test_hardware_discovery() {
        let discovery = DefaultHardwareDiscovery;
        let devices = discovery.discover_devices().unwrap();
        assert!(devices.is_empty()); // Mock implementation returns empty list

        assert!(!discovery.is_device_available("test_device"));
        assert!(discovery.get_device_info("test_device").is_none());
    }

    #[test]
    fn test_plugin_metadata() {
        let metadata = PluginMetadata {
            name: "test_plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Test Author".to_string(),
            description: "Test plugin".to_string(),
            license: "MIT".to_string(),
            min_ronn_version: "0.1.0".to_string(),
            supported_hardware: vec!["TestHW".to_string()],
            abi_version: 1,
            plugin_path: PathBuf::from("/test/plugin.so"),
            status: PluginStatus::Loaded,
        };

        assert_eq!(metadata.name, "test_plugin");
        assert_eq!(metadata.abi_version, 1);
        assert_eq!(metadata.status, PluginStatus::Loaded);
    }
}
