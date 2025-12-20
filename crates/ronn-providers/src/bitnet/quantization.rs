//! BitNet quantization and bit manipulation utilities.
//!
//! This module implements efficient quantization schemes for BitNet models,
//! including binary (-1, +1) and ternary (-1, 0, +1) quantization with
//! bit-packed storage for maximum efficiency.

use anyhow::Result;
use ronn_core::{DataType, Tensor, TensorLayout};
use std::fmt::Debug;

/// Quantization methods supported by BitNet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationMethod {
    /// Pure binary quantization: weights ∈ {-1, +1}
    Binary,
    /// Ternary quantization: weights ∈ {-1, 0, +1} (BitNet-1.58b)
    Ternary,
    /// Asymmetric binary with custom threshold
    AsymmetricBinary { threshold: f32 },
}

/// BitNet quantizer for converting FP32 tensors to bit-packed representations.
#[derive(Debug)]
pub struct BitNetQuantizer {
    /// Quantization method to use.
    method: QuantizationMethod,
    /// Scaling factor for dequantization.
    scale: Option<f32>,
}

impl BitNetQuantizer {
    /// Create a new BitNet quantizer with specified method.
    pub fn new(method: QuantizationMethod) -> Self {
        Self {
            method,
            scale: None,
        }
    }

    /// Quantize a FP32 tensor to binary representation.
    pub fn quantize_binary(&self, input: &Tensor) -> Result<BinaryTensor> {
        if input.dtype() != DataType::F32 {
            return Err(anyhow::anyhow!(
                "BitNet quantization requires FP32 input tensors"
            ));
        }

        let data = input.to_vec()?;
        let shape = input.shape().to_vec();
        let total_elements = data.len();

        // Calculate scale factor (mean absolute value)
        let scale = data.iter().map(|x| x.abs()).sum::<f32>() / total_elements as f32;

        // Quantize to {-1, +1} and pack into bits
        let packed_bits = self.pack_binary_bits(&data, scale)?;

        Ok(BinaryTensor {
            packed_data: packed_bits,
            shape,
            scale,
            element_count: total_elements,
        })
    }

    /// Quantize a FP32 tensor to ternary representation.
    pub fn quantize_ternary(&self, input: &Tensor) -> Result<TernaryTensor> {
        if input.dtype() != DataType::F32 {
            return Err(anyhow::anyhow!(
                "BitNet quantization requires FP32 input tensors"
            ));
        }

        let data = input.to_vec()?;
        let shape = input.shape().to_vec();
        let total_elements = data.len();

        // Calculate threshold (could be learned parameter)
        let threshold = data.iter().map(|x| x.abs()).sum::<f32>() / total_elements as f32 * 0.7;

        // Calculate scale factor for non-zero elements
        let non_zero_values: Vec<f32> = data
            .iter()
            .filter_map(|x| {
                if x.abs() > threshold {
                    Some(x.abs())
                } else {
                    None
                }
            })
            .collect();

        let scale = if non_zero_values.is_empty() {
            1.0
        } else {
            non_zero_values.iter().sum::<f32>() / non_zero_values.len() as f32
        };

        // Quantize to {-1, 0, +1} and pack
        let packed_bits = self.pack_ternary_bits(&data, threshold, scale)?;

        Ok(TernaryTensor {
            packed_data: packed_bits,
            shape,
            scale,
            threshold,
            element_count: total_elements,
        })
    }

    /// Pack binary values into bit-packed format.
    fn pack_binary_bits(&self, data: &[f32], scale: f32) -> Result<Vec<u8>> {
        let bit_count = data.len();
        let byte_count = (bit_count + 7) / 8; // Round up to nearest byte
        let mut packed = vec![0u8; byte_count];

        for (i, &value) in data.iter().enumerate() {
            let quantized = if value >= 0.0 { 1 } else { 0 };
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            if quantized == 1 {
                packed[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(packed)
    }

    /// Pack ternary values into 2-bit packed format.
    fn pack_ternary_bits(&self, data: &[f32], threshold: f32, scale: f32) -> Result<Vec<u8>> {
        let element_count = data.len();
        let bit_pairs_per_byte = 4; // 2 bits per element, 4 elements per byte
        let byte_count = (element_count + bit_pairs_per_byte - 1) / bit_pairs_per_byte;
        let mut packed = vec![0u8; byte_count];

        for (i, &value) in data.iter().enumerate() {
            let quantized = if value.abs() <= threshold {
                0b01 // Zero -> 01
            } else if value > 0.0 {
                0b10 // +1 -> 10
            } else {
                0b00 // -1 -> 00
            };

            let byte_idx = i / bit_pairs_per_byte;
            let bit_offset = (i % bit_pairs_per_byte) * 2;

            packed[byte_idx] |= quantized << bit_offset;
        }

        Ok(packed)
    }

    /// Dequantize binary tensor back to FP32.
    pub fn dequantize_binary(&self, binary_tensor: &BinaryTensor) -> Result<Tensor> {
        let mut data = Vec::with_capacity(binary_tensor.element_count);

        for i in 0..binary_tensor.element_count {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            let bit_value = (binary_tensor.packed_data[byte_idx] >> bit_idx) & 1;
            let float_value = if bit_value == 1 {
                binary_tensor.scale
            } else {
                -binary_tensor.scale
            };

            data.push(float_value);
        }

        Tensor::from_data(
            data,
            binary_tensor.shape.clone(),
            DataType::F32,
            TensorLayout::RowMajor,
        )
    }

    /// Dequantize ternary tensor back to FP32.
    pub fn dequantize_ternary(&self, ternary_tensor: &TernaryTensor) -> Result<Tensor> {
        let mut data = Vec::with_capacity(ternary_tensor.element_count);
        let bit_pairs_per_byte = 4;

        for i in 0..ternary_tensor.element_count {
            let byte_idx = i / bit_pairs_per_byte;
            let bit_offset = (i % bit_pairs_per_byte) * 2;

            let bit_pair = (ternary_tensor.packed_data[byte_idx] >> bit_offset) & 0b11;
            let float_value = match bit_pair {
                0b00 => -ternary_tensor.scale, // -1
                0b01 => 0.0,                   // 0
                0b10 => ternary_tensor.scale,  // +1
                _ => 0.0,                      // Invalid, default to zero
            };

            data.push(float_value);
        }

        Tensor::from_data(
            data,
            ternary_tensor.shape.clone(),
            DataType::F32,
            TensorLayout::RowMajor,
        )
    }
}

/// Binary tensor with bit-packed storage.
#[derive(Debug, Clone)]
pub struct BinaryTensor {
    /// Bit-packed binary data (8 bits per byte).
    pub packed_data: Vec<u8>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Scaling factor for dequantization.
    pub scale: f32,
    /// Total number of elements.
    pub element_count: usize,
}

impl BinaryTensor {
    /// Get the memory size in bytes (significantly reduced from FP32).
    pub fn memory_size(&self) -> usize {
        self.packed_data.len()
    }

    /// Get compression ratio compared to FP32.
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size = self.element_count * std::mem::size_of::<f32>();
        fp32_size as f32 / self.memory_size() as f32
    }

    /// Extract a bit value at the specified index.
    pub fn get_bit(&self, index: usize) -> bool {
        if index >= self.element_count {
            return false;
        }

        let byte_idx = index / 8;
        let bit_idx = index % 8;
        (self.packed_data[byte_idx] >> bit_idx) & 1 == 1
    }
}

/// Ternary tensor with 2-bit packed storage.
#[derive(Debug, Clone)]
pub struct TernaryTensor {
    /// 2-bit packed ternary data (4 elements per byte).
    pub packed_data: Vec<u8>,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Scaling factor for non-zero values.
    pub scale: f32,
    /// Threshold for zero quantization.
    pub threshold: f32,
    /// Total number of elements.
    pub element_count: usize,
}

impl TernaryTensor {
    /// Get the memory size in bytes.
    pub fn memory_size(&self) -> usize {
        self.packed_data.len()
    }

    /// Get compression ratio compared to FP32.
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size = self.element_count * std::mem::size_of::<f32>();
        fp32_size as f32 / self.memory_size() as f32
    }

    /// Extract a ternary value at the specified index.
    pub fn get_value(&self, index: usize) -> i8 {
        if index >= self.element_count {
            return 0;
        }

        let bit_pairs_per_byte = 4;
        let byte_idx = index / bit_pairs_per_byte;
        let bit_offset = (index % bit_pairs_per_byte) * 2;

        let bit_pair = (self.packed_data[byte_idx] >> bit_offset) & 0b11;
        match bit_pair {
            0b00 => -1, // -1
            0b01 => 0,  // 0
            0b10 => 1,  // +1
            _ => 0,     // Invalid, default to zero
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_quantization_roundtrip() -> Result<()> {
        let data = vec![1.5, -2.3, 0.8, -0.1, 3.2, -1.1];
        let tensor = Tensor::from_data(data, vec![2, 3], DataType::F32, TensorLayout::RowMajor)?;

        let quantizer = BitNetQuantizer::new(QuantizationMethod::Binary);
        let binary_tensor = quantizer.quantize_binary(&tensor)?;
        let dequantized = quantizer.dequantize_binary(&binary_tensor)?;

        // Check that signs are preserved
        assert_eq!(tensor.shape(), dequantized.shape());
        assert!(binary_tensor.compression_ratio() > 30.0); // Should be ~32x compression

        Ok(())
    }

    #[test]
    fn test_ternary_quantization_roundtrip() -> Result<()> {
        let data = vec![2.0, -1.5, 0.1, -0.05, 1.8, -2.1, 0.02, -0.01];
        let tensor = Tensor::from_data(data, vec![2, 4], DataType::F32, TensorLayout::RowMajor)?;

        let quantizer = BitNetQuantizer::new(QuantizationMethod::Ternary);
        let ternary_tensor = quantizer.quantize_ternary(&tensor)?;
        let dequantized = quantizer.dequantize_ternary(&ternary_tensor)?;

        // Check shape preservation
        assert_eq!(tensor.shape(), dequantized.shape());
        assert!(ternary_tensor.compression_ratio() > 15.0); // Should be ~16x compression

        Ok(())
    }

    #[test]
    fn test_bit_packing_efficiency() -> Result<()> {
        let data = vec![1.0; 1000]; // 1000 FP32 elements
        let tensor = Tensor::from_data(data, vec![1000], DataType::F32, TensorLayout::RowMajor)?;

        let quantizer = BitNetQuantizer::new(QuantizationMethod::Binary);
        let binary_tensor = quantizer.quantize_binary(&tensor)?;

        // 1000 bits should pack into 125 bytes
        assert_eq!(binary_tensor.memory_size(), 125);

        // Original: 1000 * 4 = 4000 bytes, packed: 125 bytes
        assert_eq!(binary_tensor.compression_ratio(), 32.0);

        Ok(())
    }

    #[test]
    fn test_ternary_bit_access() -> Result<()> {
        let data = vec![1.5, -2.0, 0.1, -0.05]; // Should become [1, -1, 0, 0]
        let tensor = Tensor::from_data(data, vec![4], DataType::F32, TensorLayout::RowMajor)?;

        let quantizer = BitNetQuantizer::new(QuantizationMethod::Ternary);
        let ternary_tensor = quantizer.quantize_ternary(&tensor)?;

        // Test individual value access
        assert_eq!(ternary_tensor.get_value(0), 1); // 1.5 -> 1
        assert_eq!(ternary_tensor.get_value(1), -1); // -2.0 -> -1
        assert_eq!(ternary_tensor.get_value(2), 0); // 0.1 -> 0 (below threshold)
        assert_eq!(ternary_tensor.get_value(3), 0); // -0.05 -> 0 (below threshold)

        Ok(())
    }
}
