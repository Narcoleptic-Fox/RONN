//! GGUF file format type definitions.

/// GGUF magic number: "GGUF" as little-endian u32.
pub const GGUF_MAGIC: u32 = 0x46475547;

/// Supported GGUF versions.
pub const GGUF_VERSION_2: u32 = 2;
pub const GGUF_VERSION_3: u32 = 3;

/// Default alignment for tensor data.
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// GGUF metadata value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// A parsed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    /// Try to extract as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GGUFValue::Uint32(v) => Some(*v),
            GGUFValue::Int32(v) => Some(*v as u32),
            GGUFValue::Uint8(v) => Some(*v as u32),
            GGUFValue::Uint16(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to extract as u64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GGUFValue::Uint64(v) => Some(*v),
            GGUFValue::Uint32(v) => Some(*v as u64),
            GGUFValue::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Try to extract as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GGUFValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GGUFValue::Float32(v) => Some(*v),
            GGUFValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to extract as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GGUFValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Descriptor for a tensor stored in the GGUF file.
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    /// Tensor name (e.g., "blk.0.attn_q.weight").
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Shape dimensions.
    pub dims: Vec<u64>,
    /// Quantization type.
    pub dtype: nnx_quant::GGMLType,
    /// Byte offset into the tensor data section.
    pub offset: u64,
}

impl GGUFTensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> u64 {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }
}
