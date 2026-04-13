use crate::error::{OptimizationError, Result};
use crate::passes::*;
use ronn_core::ModelGraph;
use std::collections::HashMap;
use tracing::{debug, info};

/// Optimization levels similar to compiler optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimizations
    O0,
    /// Basic optimizations (constant folding, dead code elimination)
    O1,
    /// Standard optimizations (O1 + node fusion, layout optimization)
    O2,
    /// Aggressive optimizations (O2 + provider-specific passes)
    O3,
}

/// Main optimizer that applies optimization passes to a graph
pub struct Optimizer {
    pass_manager: PassManager,
    level: OptimizationLevel,
}

impl Optimizer {
    /// Create a new optimizer with the specified optimization level
    pub fn new(level: OptimizationLevel) -> Self {
        let mut pass_manager = PassManager::new();

        // Register passes based on optimization level
        match level {
            OptimizationLevel::O0 => {
                // No optimizations
            }
            OptimizationLevel::O1 => {
                pass_manager.add_pass(Box::new(ConstantFoldingPass));
                pass_manager.add_pass(Box::new(DeadCodeEliminationPass));
            }
            OptimizationLevel::O2 => {
                pass_manager.add_pass(Box::new(ConstantFoldingPass));
                pass_manager.add_pass(Box::new(DeadCodeEliminationPass));
                pass_manager.add_pass(Box::new(NodeFusionPass));
                pass_manager.add_pass(Box::new(LayoutOptimizationPass));
            }
            OptimizationLevel::O3 => {
                pass_manager.add_pass(Box::new(ConstantFoldingPass));
                pass_manager.add_pass(Box::new(DeadCodeEliminationPass));
                pass_manager.add_pass(Box::new(NodeFusionPass));
                pass_manager.add_pass(Box::new(LayoutOptimizationPass));
                pass_manager.add_pass(Box::new(CpuOptimizationPass));
                pass_manager.add_pass(Box::new(GpuOptimizationPass));
                pass_manager.add_pass(Box::new(SparsityOptimizationPass));
            }
        }

        Self {
            pass_manager,
            level,
        }
    }

    /// Optimize a model graph
    pub fn optimize(&self, graph: &mut ModelGraph) -> Result<OptimizationStats> {
        info!("Starting optimization with level {:?}", self.level);
        self.pass_manager.run(graph)
    }

    /// Get the number of registered passes
    pub fn pass_count(&self) -> usize {
        self.pass_manager.pass_count()
    }

    /// Get the optimization level
    pub fn level(&self) -> OptimizationLevel {
        self.level
    }
}

/// Manages and executes optimization passes
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl PassManager {
    /// Create a new pass manager
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add an optimization pass
    pub fn add_pass(&mut self, pass: Box<dyn OptimizationPass>) {
        self.passes.push(pass);
    }

    /// Run all passes on the graph
    pub fn run(&self, graph: &mut ModelGraph) -> Result<OptimizationStats> {
        let mut stats = OptimizationStats::new();
        let mut modified = true;
        let mut iteration = 0;
        const MAX_ITERATIONS: usize = 10;

        // Run passes iteratively until no more changes or max iterations reached
        while modified && iteration < MAX_ITERATIONS {
            modified = false;
            iteration += 1;

            debug!("Optimization iteration {}", iteration);

            for pass in &self.passes {
                let pass_name = pass.name();
                debug!("Running pass: {}", pass_name);

                let result = pass.run(graph).map_err(|e| OptimizationError::PassFailed {
                    pass_name: pass_name.to_string(),
                    reason: e.to_string(),
                })?;

                if result.nodes_removed > 0 || result.nodes_fused > 0 || result.nodes_modified > 0 {
                    modified = true;
                    stats.merge(result.clone());
                }

                info!(
                    "Pass {} completed: {} nodes removed, {} fused, {} modified",
                    pass_name, result.nodes_removed, result.nodes_fused, result.nodes_modified
                );
            }
        }

        stats.iterations = iteration;
        info!(
            "Optimization completed after {} iterations: {} total changes",
            stats.iterations,
            stats.total_changes()
        );

        Ok(stats)
    }

    /// Get the number of passes
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from running optimization passes
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    pub nodes_removed: usize,
    pub nodes_fused: usize,
    pub nodes_modified: usize,
    pub iterations: usize,
    pub pass_stats: HashMap<String, PassStats>,
}

impl OptimizationStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn merge(&mut self, other: PassStats) {
        self.nodes_removed += other.nodes_removed;
        self.nodes_fused += other.nodes_fused;
        self.nodes_modified += other.nodes_modified;
    }

    pub fn total_changes(&self) -> usize {
        self.nodes_removed + self.nodes_fused + self.nodes_modified
    }
}

/// Statistics from a single pass execution
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    pub nodes_removed: usize,
    pub nodes_fused: usize,
    pub nodes_modified: usize,
}
