//! Shared activation quantization utilities.
//!
//! Provides a common Q8_0 round-trip helper used by FFN and attention layers.

use crate::config::ActivationQuantization;
use nnx_core::error::{EngineError, Result};

/// Apply optional activation quantization to a buffer in-place.
///
/// Currently supports Q8_0 round-trip (quantize + dequantize) which
/// reduces numerical precision while maintaining the same memory footprint.
///
/// The round-trip preserves f32 output for downstream kernels while simulating
/// quantization error that would occur in a true quantized execution path.
///
/// # Arguments
/// * `activations` - Buffer to quantize in-place (modified)
/// * `rows` - Number of rows in the matrix view
/// * `cols` - Number of columns in the matrix view  
/// * `mode` - Quantization mode (None or Q8_0)
pub(crate) fn maybe_quantize_activations(
    activations: &mut [f32],
    rows: usize,
    cols: usize,
    mode: ActivationQuantization,
) -> Result<()> {
    match mode {
        ActivationQuantization::None => Ok(()),
        ActivationQuantization::Q8_0 => {
            nnx_quant::encode::roundtrip_matrix_in_place(
                activations,
                rows,
                cols,
                nnx_quant::GGMLType::Q8_0,
            )
            .map_err(EngineError::Quantization)
        }
    }
}
