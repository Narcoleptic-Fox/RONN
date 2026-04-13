// Optimization passes for graph transformation

mod constant_folding;
mod dead_code;
mod fusion;
mod layout;
mod provider_specific;
mod sparsity;

pub use constant_folding::ConstantFoldingPass;
pub use dead_code::DeadCodeEliminationPass;
pub use fusion::NodeFusionPass;
pub use layout::LayoutOptimizationPass;
pub use provider_specific::{CpuOptimizationPass, GpuOptimizationPass};
pub use sparsity::SparsityOptimizationPass;

use crate::error::Result;
use crate::optimizer::PassStats;
use ronn_core::ModelGraph;

/// Trait for optimization passes
pub trait OptimizationPass: Send + Sync {
    /// Get the name of the pass
    fn name(&self) -> &str;

    /// Run the optimization pass on a graph
    fn run(&self, graph: &mut ModelGraph) -> Result<PassStats>;

    /// Check if the pass should run (can be overridden for conditional passes)
    fn should_run(&self, _graph: &ModelGraph) -> bool {
        true
    }
}
