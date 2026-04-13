//! Quantization block structures.
//!
//! Each block type represents a packed group of quantized values.
//! Block sizes and layouts match the GGML specification exactly.

/// Q4_0: 32 values quantized to 4 bits, one f16 scale factor.
/// Block: 2 bytes (scale) + 16 bytes (32 nibbles) = 18 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub scale: half::f16,
    pub quants: [u8; 16],
}

/// Q8_0: 32 values quantized to 8 bits, one f16 scale factor.
/// Block: 2 bytes (scale) + 32 bytes (32 × i8) = 34 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub scale: half::f16,
    pub quants: [i8; 32],
}

/// Q4_K: k-quant 4-bit, block size 256, with sub-block scales.
/// Most popular format for quality/size tradeoff.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4K {
    pub d: half::f16,       // super-block scale
    pub dmin: half::f16,    // super-block min
    pub scales: [u8; 12],   // sub-block scales (packed)
    pub quants: [u8; 128],  // 256 values at 4 bits each
}

/// Q5_K: k-quant 5-bit, block size 256.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    pub d: half::f16,
    pub dmin: half::f16,
    pub scales: [u8; 12],
    pub qh: [u8; 32],      // high bits
    pub qs: [u8; 128],     // low 4 bits
}

/// Q6_K: k-quant 6-bit, block size 256.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; 128],     // low 4 bits
    pub qh: [u8; 64],      // high 2 bits
    pub scales: [i8; 16],  // sub-block scales
    pub d: half::f16,      // super-block scale
}

// Compile-time size assertions
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);
const _: () = assert!(std::mem::size_of::<BlockQ5K>() == 176);
const _: () = assert!(std::mem::size_of::<BlockQ6K>() == 210);
