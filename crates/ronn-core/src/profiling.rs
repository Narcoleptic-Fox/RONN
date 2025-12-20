//! Performance profiling infrastructure
//!
//! Provides detailed profiling of inference operations to identify bottlenecks.
//! Minimal overhead when disabled, detailed insights when enabled.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Enable profiling
    pub enabled: bool,
    /// Profile individual operators
    pub profile_ops: bool,
    /// Profile memory allocations
    pub profile_memory: bool,
    /// Profile data transfers
    pub profile_transfers: bool,
    /// Minimum duration to record (filter noise)
    pub min_duration_us: u64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            profile_ops: true,
            profile_memory: true,
            profile_transfers: true,
            min_duration_us: 10, // 10 microseconds
        }
    }
}

impl ProfileConfig {
    /// Create a development profiling config (everything enabled)
    pub fn development() -> Self {
        Self {
            enabled: true,
            profile_ops: true,
            profile_memory: true,
            profile_transfers: true,
            min_duration_us: 1,
        }
    }

    /// Create a production profiling config (minimal overhead)
    pub fn production() -> Self {
        Self {
            enabled: true,
            profile_ops: true,
            profile_memory: false,
            profile_transfers: false,
            min_duration_us: 100, // Only record slow ops
        }
    }
}

/// A single profiling event
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    /// Event name
    pub name: String,
    /// Event category (op, memory, transfer, etc.)
    pub category: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Start timestamp
    pub timestamp: Instant,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ProfileEvent {
    /// Create a new profile event
    pub fn new(name: String, category: String, duration: Duration) -> Self {
        Self {
            name,
            category,
            duration_us: duration.as_micros() as u64,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Profiler for recording performance events
pub struct Profiler {
    config: ProfileConfig,
    events: Arc<Mutex<Vec<ProfileEvent>>>,
    session_start: Instant,
}

impl Profiler {
    /// Create a new profiler
    pub fn new(config: ProfileConfig) -> Self {
        Self {
            config,
            events: Arc::new(Mutex::new(Vec::new())),
            session_start: Instant::now(),
        }
    }

    /// Create a profiler with default config
    pub fn default() -> Self {
        Self::new(ProfileConfig::default())
    }

    /// Start profiling a named operation
    ///
    /// # Returns
    ///
    /// A ProfileScope that automatically records duration on drop
    pub fn scope(&self, name: impl Into<String>, category: impl Into<String>) -> ProfileScope {
        ProfileScope::new(self.clone(), name.into(), category.into())
    }

    /// Record an event
    pub fn record(&self, event: ProfileEvent) {
        if !self.config.enabled {
            return;
        }

        if event.duration_us < self.config.min_duration_us {
            return;
        }

        let mut events = self.events.lock().unwrap();
        events.push(event);
    }

    /// Get all recorded events
    pub fn events(&self) -> Vec<ProfileEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Clear all events
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    /// Generate profiling report
    pub fn report(&self) -> ProfileReport {
        let events = self.events();
        ProfileReport::from_events(events, self.session_start.elapsed())
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

impl Clone for Profiler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            events: Arc::clone(&self.events),
            session_start: self.session_start,
        }
    }
}

/// RAII scope for automatic profiling
///
/// Records duration automatically when dropped
pub struct ProfileScope {
    profiler: Profiler,
    name: String,
    category: String,
    start: Instant,
    metadata: HashMap<String, String>,
}

impl ProfileScope {
    fn new(profiler: Profiler, name: String, category: String) -> Self {
        Self {
            profiler,
            name,
            category,
            start: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this scope
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let mut event = ProfileEvent::new(self.name.clone(), self.category.clone(), duration);
        event.metadata = self.metadata.clone();
        self.profiler.record(event);
    }
}

/// Profiling report with aggregated statistics
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Total session duration
    pub total_duration: Duration,
    /// Events grouped by category
    pub by_category: HashMap<String, CategoryStats>,
    /// Events grouped by name
    pub by_name: HashMap<String, OperationStats>,
    /// All events
    pub events: Vec<ProfileEvent>,
}

/// Statistics for a category
#[derive(Debug, Clone, serde::Serialize)]
pub struct CategoryStats {
    /// Number of events
    pub count: usize,
    /// Total time spent
    pub total_us: u64,
    /// Average time per event
    pub avg_us: u64,
    /// Minimum time
    pub min_us: u64,
    /// Maximum time
    pub max_us: u64,
    /// Percentage of total time
    pub percentage: f64,
}

/// Statistics for a specific operation
#[derive(Debug, Clone, serde::Serialize)]
pub struct OperationStats {
    /// Number of calls
    pub count: usize,
    /// Total time
    pub total_us: u64,
    /// Average time
    pub avg_us: u64,
    /// Minimum time
    pub min_us: u64,
    /// Maximum time
    pub max_us: u64,
    /// Standard deviation
    pub std_dev_us: f64,
}

impl ProfileReport {
    /// Create report from events
    pub fn from_events(events: Vec<ProfileEvent>, total_duration: Duration) -> Self {
        let total_us = total_duration.as_micros() as u64;

        // Group by category
        let mut by_category: HashMap<String, Vec<u64>> = HashMap::new();
        for event in &events {
            by_category
                .entry(event.category.clone())
                .or_insert_with(Vec::new)
                .push(event.duration_us);
        }

        let category_stats: HashMap<String, CategoryStats> = by_category
            .into_iter()
            .map(|(cat, durations)| {
                let count = durations.len();
                let total: u64 = durations.iter().sum();
                let min = *durations.iter().min().unwrap_or(&0);
                let max = *durations.iter().max().unwrap_or(&0);
                let avg = if count > 0 { total / count as u64 } else { 0 };
                let percentage = if total_us > 0 {
                    (total as f64 / total_us as f64) * 100.0
                } else {
                    0.0
                };

                (
                    cat,
                    CategoryStats {
                        count,
                        total_us: total,
                        avg_us: avg,
                        min_us: min,
                        max_us: max,
                        percentage,
                    },
                )
            })
            .collect();

        // Group by name
        let mut by_name: HashMap<String, Vec<u64>> = HashMap::new();
        for event in &events {
            by_name
                .entry(event.name.clone())
                .or_insert_with(Vec::new)
                .push(event.duration_us);
        }

        let name_stats: HashMap<String, OperationStats> = by_name
            .into_iter()
            .map(|(name, durations)| {
                let count = durations.len();
                let total: u64 = durations.iter().sum();
                let min = *durations.iter().min().unwrap_or(&0);
                let max = *durations.iter().max().unwrap_or(&0);
                let avg = if count > 0 { total / count as u64 } else { 0 };

                // Calculate standard deviation
                let variance: f64 = if count > 1 {
                    durations
                        .iter()
                        .map(|&d| {
                            let diff = d as f64 - avg as f64;
                            diff * diff
                        })
                        .sum::<f64>()
                        / (count - 1) as f64
                } else {
                    0.0
                };
                let std_dev = variance.sqrt();

                (
                    name,
                    OperationStats {
                        count,
                        total_us: total,
                        avg_us: avg,
                        min_us: min,
                        max_us: max,
                        std_dev_us: std_dev,
                    },
                )
            })
            .collect();

        Self {
            total_duration,
            by_category: category_stats,
            by_name: name_stats,
            events,
        }
    }

    /// Print a human-readable report
    pub fn print(&self) {
        println!("\n=== Profiling Report ===");
        println!(
            "Total Duration: {:.2}ms\n",
            self.total_duration.as_secs_f64() * 1000.0
        );

        println!("By Category:");
        let mut categories: Vec<_> = self.by_category.iter().collect();
        categories.sort_by(|a, b| b.1.total_us.cmp(&a.1.total_us));
        for (cat, stats) in categories {
            println!(
                "  {}: {:.2}ms ({:.1}%) - {} calls, avg {:.2}µs",
                cat,
                stats.total_us as f64 / 1000.0,
                stats.percentage,
                stats.count,
                stats.avg_us
            );
        }

        println!("\nTop 10 Operations:");
        let mut operations: Vec<_> = self.by_name.iter().collect();
        operations.sort_by(|a, b| b.1.total_us.cmp(&a.1.total_us));
        for (name, stats) in operations.iter().take(10) {
            println!(
                "  {}: {:.2}ms - {} calls, avg {:.2}µs ± {:.2}µs",
                name,
                stats.total_us as f64 / 1000.0,
                stats.count,
                stats.avg_us,
                stats.std_dev_us
            );
        }

        println!("\n=== End Report ===\n");
    }

    /// Export report as JSON
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "total_duration_ms": self.total_duration.as_secs_f64() * 1000.0,
            "by_category": self.by_category,
            "by_name": self.by_name,
            "event_count": self.events.len(),
        })
    }
}

/// Global profiler instance
static GLOBAL_PROFILER: std::sync::OnceLock<Profiler> = std::sync::OnceLock::new();

/// Initialize global profiler
pub fn init_profiler(config: ProfileConfig) {
    let _ = GLOBAL_PROFILER.set(Profiler::new(config));
}

/// Get global profiler
pub fn global_profiler() -> &'static Profiler {
    GLOBAL_PROFILER.get_or_init(Profiler::default)
}

/// Profile a scope with the global profiler
#[macro_export]
macro_rules! profile {
    ($name:expr, $category:expr) => {
        let _scope = $crate::profiling::global_profiler().scope($name, $category);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new(ProfileConfig::default());
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_profile_scope() {
        let profiler = Profiler::new(ProfileConfig::development());
        {
            let _scope = profiler.scope("test_op", "test");
            thread::sleep(Duration::from_millis(10));
        }

        let events = profiler.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test_op");
        assert!(events[0].duration_us >= 10_000); // At least 10ms
    }

    #[test]
    fn test_profiler_report() {
        let profiler = Profiler::new(ProfileConfig::development());

        // Record some events
        for i in 0..5 {
            let _scope = profiler.scope(format!("op_{}", i), "ops");
            thread::sleep(Duration::from_millis(1));
        }

        let report = profiler.report();
        assert_eq!(report.events.len(), 5);
        assert!(report.by_category.contains_key("ops"));
        assert_eq!(report.by_category["ops"].count, 5);
    }

    #[test]
    fn test_min_duration_filter() {
        let config = ProfileConfig {
            enabled: true,
            min_duration_us: 1000, // 1ms minimum
            ..Default::default()
        };
        let profiler = Profiler::new(config);

        // Fast operation (should be filtered)
        {
            let _scope = profiler.scope("fast", "test");
            // No sleep - very fast
        }

        // Slow operation (should be recorded)
        {
            let _scope = profiler.scope("slow", "test");
            thread::sleep(Duration::from_millis(2));
        }

        let events = profiler.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "slow");
    }

    #[test]
    fn test_report_statistics() {
        let profiler = Profiler::new(ProfileConfig::development());

        for _ in 0..10 {
            let _scope = profiler.scope("test_op", "test");
            thread::sleep(Duration::from_millis(1));
        }

        let report = profiler.report();
        let stats = &report.by_name["test_op"];

        assert_eq!(stats.count, 10);
        assert!(stats.avg_us >= 1000); // At least 1ms average
        assert!(stats.min_us <= stats.max_us);
    }
}
