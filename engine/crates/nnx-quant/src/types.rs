//! GGML quantization type enumeration and metadata.

/// GGML quantization types — determines how tensor data is packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, // deprecated
    // Q4_3 = 5, // deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GGMLType {
    /// Try to convert from a raw u32 value.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Block size (number of elements per quantization block).
    pub fn block_numel(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S | Self::IQ3XXS | Self::IQ3S => 256,
            Self::IQ1S | Self::IQ1M => 256,
            Self::IQ4NL | Self::IQ4XS => 32,
        }
    }

    /// Size of one quantization block in bytes.
    pub fn block_size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,              // 2 (f16 scale) + 16 (4-bit × 32)
            Self::Q4_1 => 20,              // 2 (scale) + 2 (min) + 16
            Self::Q5_0 => 22,              // 2 + 4 (high bits) + 16
            Self::Q5_1 => 24,              // 2 + 2 + 4 + 16
            Self::Q8_0 => 34,              // 2 (scale) + 32 (8-bit × 32)
            Self::Q8_1 => 40,              // 4 (f32 scale) + 4 (f32 min) + 32
            Self::Q2K => 256 / 4 + 16 + 1, // approximate
            Self::Q3K => 256 * 3 / 8 + 12 + 2,
            Self::Q4K => 144, // 2 + 2 + 12 + 128
            Self::Q5K => 176, // 2 + 2 + 12 + 160
            Self::Q6K => 210, // 2 + 128 + 64 + 16
            Self::Q8K => 292, // 4 + 256 + 32
            // IQ types — sizes vary, these are approximate
            _ => 0, // TODO: fill in IQ block sizes
        }
    }

    /// Whether this is a quantized (non-native) type.
    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3S => "IQ3_S",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ4XS => "IQ4_XS",
        }
    }
}

impl std::fmt::Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
