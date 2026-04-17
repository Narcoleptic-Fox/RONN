# RONN-Llama: Native Llama Model Support via GGUF

## Branch: `feature/ronn-llama`

## Overview

Add a `ronn-llama` crate to RONN that provides native Llama model loading and inference, bypassing ONNX entirely. This crate loads GGUF files (the standard distribution format for local LLM inference), implements the Llama transformer architecture in pure Rust using RONN's existing tensor operations, and provides a text generation API with sampling, KV caching, and streaming output.

This is the foundation that all RONN optimization features (speculative decoding, KV cache compression, early exit, activation sparsity, continuous batching) will build upon.

## Why GGUF, Not ONNX

- **Ecosystem alignment**: Every quantized Llama model on HuggingFace ships as GGUF. Nobody distributes Llama as ONNX.
- **Quantization built-in**: GGUF includes quantization metadata and dequantization logic. ONNX would require separate quantization handling.
- **Tokenizer included**: GGUF embeds the tokenizer vocabulary and merge rules. No separate tokenizer files needed.
- **Architecture metadata**: GGUF contains all hyperparameters (head counts, layer counts, context length, RoPE settings) in structured key-value pairs.
- **Memory-mapped loading**: GGUF is designed for mmap — tensor data is aligned and can be loaded without copying.

## Architecture Context

This crate sits alongside `ronn-onnx` as an alternative model format loader, but with a key difference: it includes the full model architecture implementation, not just operator dispatch.

```
ronn-llama (NEW)
├── gguf/          # GGUF file format parser
│   ├── parser.rs  # Header, metadata, tensor info parsing
│   ├── types.rs   # GGUF type definitions and constants
│   └── mmap.rs    # Memory-mapped tensor data access
├── model/         # Llama architecture implementation
│   ├── config.rs  # Model configuration from GGUF metadata
│   ├── layers.rs  # RMSNorm, RoPE, SwiGLU, Attention
│   ├── block.rs   # Transformer block (attention + FFN)
│   └── llama.rs   # Full model: embedding → N blocks → head
├── tokenizer/     # BPE tokenizer from GGUF vocab
│   ├── bpe.rs     # Byte-pair encoding implementation
│   └── vocab.rs   # Vocabulary and special tokens
├── cache/         # KV cache management
│   └── kv.rs      # Per-layer KV cache with position tracking
├── generate/      # Text generation
│   ├── sampler.rs # Temperature, top-p, top-k, repetition penalty
│   └── stream.rs  # Token-by-token streaming generation
├── lib.rs         # Public API
└── error.rs       # Error types
```

Integration points with existing RONN crates:
- **ronn-core**: `Tensor`, `DataType`, `TensorLayout` for all tensor operations
- **ronn-providers**: CPU/GPU execution providers for compute
- **ronn-hrm**: Future integration — route tokens through System 1/2 based on generation confidence
- **candle-core**: Already a workspace dependency, used for underlying tensor math

## Dependencies

All of these are already workspace dependencies — no new external deps needed:

```toml
[dependencies]
ronn-core = { path = "../ronn-core" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
memmap2 = { workspace = true }
half = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }
tracing = { workspace = true }
serde = { workspace = true }
rayon = { workspace = true }
```

---

## Part 1: GGUF File Format Parser

### 1.1 GGUF Type Definitions (`gguf/types.rs`)

The GGUF format is fully specified. Implement these types exactly:

```rust
/// GGUF magic number: "GGUF" as little-endian u32
pub const GGUF_MAGIC: u32 = 0x46475547; // "GGUF"

/// Supported GGUF versions
pub const GGUF_VERSION_2: u32 = 2;
pub const GGUF_VERSION_3: u32 = 3;

/// GGUF metadata value types
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

/// GGUF quantization types — this is the critical enum
/// Each type defines how tensor data is packed and how to dequantize
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,    // 4-bit quantization, block size 32, 1 scale per block
    Q4_1 = 3,    // 4-bit with min, block size 32
    // Q4_2 = 4, // deprecated
    // Q4_3 = 5, // deprecated
    Q5_0 = 6,    // 5-bit quantization
    Q5_1 = 7,    // 5-bit with min
    Q8_0 = 8,    // 8-bit quantization, block size 32
    Q8_1 = 9,    // 8-bit with min
    Q2_K = 10,   // k-quant 2-bit
    Q3_K = 11,   // k-quant 3-bit
    Q4_K = 12,   // k-quant 4-bit (most common for good quality/size balance)
    Q5_K = 13,   // k-quant 5-bit
    Q6_K = 14,   // k-quant 6-bit
    Q8_K = 15,   // k-quant 8-bit
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
}
```

**Dequantization block structures** — implement at minimum Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, F16, F32, BF16. These cover ~95% of models on HuggingFace:

```rust
/// Q4_0: 32 values quantized to 4 bits each, one f16 scale factor
/// Block size: 2 + 16 = 18 bytes for 32 values
#[repr(C)]
pub struct BlockQ4_0 {
    pub scale: half::f16,        // 2 bytes
    pub quants: [u8; 16],        // 16 bytes, 2 values per byte (4 bits each)
}

/// Q8_0: 32 values quantized to 8 bits each, one f16 scale factor
/// Block size: 2 + 32 = 34 bytes for 32 values
#[repr(C)]
pub struct BlockQ8_0 {
    pub scale: half::f16,        // 2 bytes
    pub quants: [i8; 32],        // 32 bytes
}

/// Q4_K: k-quant 4-bit, block size 256, with 8 sub-blocks
/// This is the most popular quantization for quality/size tradeoff
#[repr(C)]
pub struct BlockQ4_K {
    pub d: half::f16,            // super-block scale
    pub dmin: half::f16,         // super-block min
    pub scales: [u8; 12],        // sub-block scales (packed)
    pub quants: [u8; 128],       // 256 values at 4 bits each
}
```

Each block type needs a `dequantize(block: &BlockXX, output: &mut [f32])` function that unpacks the quantized values to f32. The dequantization formulas are:

- **Q4_0**: `output[i] = scale * (quant_nibble[i] - 8)` where nibbles are extracted from packed bytes
- **Q8_0**: `output[i] = scale * quant[i]`
- **Q4_K**: More complex with super-block and sub-block scales. Reference llama.cpp's `dequantize_row_q4_K`
- **F16**: Direct `half::f16::to_f32()` conversion
- **BF16**: Direct bf16 to f32 conversion via bit manipulation

### 1.2 GGUF Parser (`gguf/parser.rs`)

Parse the GGUF file structure:

```
File Layout:
┌─────────────────────────────────────┐
│ Header                              │
│   magic: u32 (0x46475547)           │
│   version: u32 (2 or 3)            │
│   tensor_count: u64                 │
│   metadata_kv_count: u64            │
├─────────────────────────────────────┤
│ Metadata Key-Value Pairs            │
│   For each pair:                    │
│     key_length: u64                 │
│     key: [u8; key_length]           │
│     value_type: u32                 │
│     value: (varies by type)         │
├─────────────────────────────────────┤
│ Tensor Infos                        │
│   For each tensor:                  │
│     name_length: u64                │
│     name: [u8; name_length]         │
│     n_dims: u32                     │
│     dims: [u64; n_dims]             │
│     type: u32 (GGMLType)            │
│     offset: u64 (from data start)   │
├─────────────────────────────────────┤
│ Padding (align to GGUF_ALIGNMENT)   │
├─────────────────────────────────────┤
│ Tensor Data (mmap-friendly)         │
│   Each tensor aligned to 32 bytes   │
└─────────────────────────────────────┘
```

Key implementation details:

```rust
pub const GGUF_ALIGNMENT: usize = 32;

pub struct GGUFFile {
    pub header: GGUFHeader,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensor_infos: Vec<TensorInfo>,
    pub data_offset: usize, // byte offset where tensor data begins
}

pub struct GGUFHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: GGMLType,
    pub offset: u64, // relative to data_offset
}

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

impl GGUFFile {
    /// Parse a GGUF file from a byte slice (typically mmap'd)
    pub fn parse(data: &[u8]) -> Result<Self> { ... }

    /// Get a metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&GGUFValue> { ... }

    /// Get string metadata (convenience)
    pub fn get_string(&self, key: &str) -> Option<&str> { ... }

    /// Get u32 metadata (convenience)
    pub fn get_u32(&self, key: &str) -> Option<u32> { ... }

    /// Get f32 metadata (convenience)
    pub fn get_f32(&self, key: &str) -> Option<f32> { ... }

    /// Get tensor data as a byte slice
    pub fn tensor_data(&self, info: &TensorInfo, file_data: &[u8]) -> &[u8] { ... }

    /// Dequantize a tensor to f32 Vec
    pub fn dequantize_tensor(&self, info: &TensorInfo, file_data: &[u8]) -> Result<Vec<f32>> { ... }

    /// Load a tensor as a Candle tensor (dequantized to f32 or kept as f16)
    pub fn load_tensor(&self, name: &str, file_data: &[u8], device: &candle_core::Device) -> Result<candle_core::Tensor> { ... }
}
```

### 1.3 Memory-Mapped Access (`gguf/mmap.rs`)

Use `memmap2` for zero-copy file access:

```rust
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub struct MappedGGUF {
    mmap: Mmap,
    gguf: GGUFFile,
}

impl MappedGGUF {
    /// Open a GGUF file with memory mapping
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let gguf = GGUFFile::parse(&mmap)?;
        Ok(Self { mmap, gguf })
    }

    /// Load a named tensor, dequantized to the target dtype
    pub fn load_tensor(&self, name: &str, device: &candle_core::Device) -> Result<candle_core::Tensor> {
        self.gguf.load_tensor(name, &self.mmap, device)
    }

    /// Get model metadata
    pub fn metadata(&self) -> &HashMap<String, GGUFValue> {
        &self.gguf.metadata
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.gguf.tensor_infos.iter().map(|t| t.name.as_str()).collect()
    }
}
```

---

## Part 2: Llama Model Architecture

### 2.1 Model Configuration (`model/config.rs`)

Extract configuration from GGUF metadata keys. These are standardized:

```rust
pub struct LlamaConfig {
    // Architecture
    pub vocab_size: usize,          // "llama.vocab_size" or tokenizer vocab count
    pub hidden_size: usize,         // "llama.embedding_length"
    pub intermediate_size: usize,   // "llama.feed_forward_length"
    pub num_hidden_layers: usize,   // "llama.block_count"
    pub num_attention_heads: usize, // "llama.attention.head_count"
    pub num_kv_heads: usize,        // "llama.attention.head_count_kv" (for GQA)
    pub head_dim: usize,            // hidden_size / num_attention_heads
    pub max_position_embeddings: usize, // "llama.context_length"

    // RoPE configuration
    pub rope_theta: f32,            // "llama.rope.freq_base" (default: 10000.0)
    pub rope_scaling_type: Option<RoPEScalingType>, // "llama.rope.scaling.type"
    pub rope_scaling_factor: Option<f32>,            // "llama.rope.scaling.factor"

    // Normalization
    pub rms_norm_eps: f32,          // "llama.attention.layer_norm_rms_epsilon" (default: 1e-5)

    // Vocabulary
    pub bos_token_id: u32,          // "tokenizer.ggml.bos_token_id"
    pub eos_token_id: u32,          // "tokenizer.ggml.eos_token_id"
    pub pad_token_id: Option<u32>,  // "tokenizer.ggml.padding_token_id"
}

pub enum RoPEScalingType {
    Linear,     // Simple linear interpolation
    Dynamic,    // NTK-aware dynamic scaling
    YaRN,       // Yet another RoPE extensioN
    Llama3,     // Llama 3.1 style (NTK + linear blend)
}

impl LlamaConfig {
    /// Extract configuration from GGUF metadata
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self> {
        // Read all keys with "llama." prefix
        // Fall back to sensible defaults where possible
        // Error on missing critical values (hidden_size, num_layers, etc.)
        ...
    }

    /// Number of KV heads per group (for GQA)
    pub fn num_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads
    }

    /// Whether this model uses GQA (grouped query attention)
    pub fn uses_gqa(&self) -> bool {
        self.num_kv_heads < self.num_attention_heads
    }
}
```

### 2.2 GGUF Tensor Name Mapping

GGUF tensor names follow a convention. Map them to logical model components:

```rust
/// Standard GGUF tensor names for Llama models
pub struct TensorNames;

impl TensorNames {
    // Embeddings
    pub const TOKEN_EMBD: &'static str = "token_embd.weight";

    // Output
    pub const OUTPUT_NORM: &'static str = "output_norm.weight";
    pub const OUTPUT: &'static str = "output.weight"; // LM head

    // Per-layer (replace {n} with layer index)
    pub const ATTN_NORM: &'static str = "blk.{n}.attn_norm.weight";
    pub const ATTN_Q: &'static str = "blk.{n}.attn_q.weight";
    pub const ATTN_K: &'static str = "blk.{n}.attn_k.weight";
    pub const ATTN_V: &'static str = "blk.{n}.attn_v.weight";
    pub const ATTN_OUTPUT: &'static str = "blk.{n}.attn_output.weight";

    pub const FFN_NORM: &'static str = "blk.{n}.ffn_norm.weight";
    pub const FFN_GATE: &'static str = "blk.{n}.ffn_gate.weight";  // SwiGLU gate
    pub const FFN_UP: &'static str = "blk.{n}.ffn_up.weight";      // SwiGLU up
    pub const FFN_DOWN: &'static str = "blk.{n}.ffn_down.weight";  // SwiGLU down

    /// Get the tensor name for a specific layer
    pub fn layer_name(template: &str, layer_idx: usize) -> String {
        template.replace("{n}", &layer_idx.to_string())
    }
}
```

### 2.3 Layer Implementations (`model/layers.rs`)

Implement each Llama-specific layer. These differ from standard transformer layers:

#### RMSNorm (NOT LayerNorm)

Llama uses RMSNorm, which is simpler and slightly faster than LayerNorm. No bias, no mean subtraction:

```
RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
```

```rust
pub struct RMSNorm {
    weight: candle_core::Tensor,  // [hidden_size]
    eps: f32,
}

impl RMSNorm {
    pub fn new(weight: candle_core::Tensor, eps: f32) -> Self { ... }

    pub fn forward(&self, x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        // 1. Compute RMS: sqrt(mean(x^2) + eps)
        // 2. Normalize: x / rms
        // 3. Scale: normalized * weight
        //
        // IMPORTANT: compute in f32 for numerical stability even if input is f16
        // Cast back to input dtype after normalization
        ...
    }
}
```

#### Rotary Position Embeddings (RoPE)

RoPE encodes position information by rotating pairs of dimensions in Q and K:

```
For position pos, dimension pair (2i, 2i+1):
  theta_i = rope_theta ^ (-2i / head_dim)
  freq = pos * theta_i

  q_rotated[2i]   = q[2i] * cos(freq) - q[2i+1] * sin(freq)
  q_rotated[2i+1] = q[2i] * sin(freq) + q[2i+1] * cos(freq)
```

```rust
pub struct RotaryEmbedding {
    cos_cache: candle_core::Tensor,  // [max_seq_len, head_dim/2]
    sin_cache: candle_core::Tensor,  // [max_seq_len, head_dim/2]
    head_dim: usize,
}

impl RotaryEmbedding {
    /// Pre-compute sin/cos tables for all positions up to max_seq_len
    pub fn new(config: &LlamaConfig, device: &candle_core::Device) -> Result<Self> {
        let head_dim = config.head_dim;
        let max_pos = config.max_position_embeddings;
        let theta = config.rope_theta;

        // Compute frequency bands: theta^(-2i/head_dim) for i in 0..head_dim/2
        // Then compute position * frequency for all positions
        // Cache cos and sin values

        // Handle RoPE scaling if configured (Llama 3.1 extended context)
        ...
    }

    /// Apply rotary embeddings to query and key tensors
    /// q, k shape: [batch, seq_len, num_heads, head_dim]
    /// start_pos: position offset for incremental decoding
    pub fn apply(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        start_pos: usize,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
        // Extract cos/sin for positions [start_pos..start_pos+seq_len]
        // Split q and k into even/odd pairs
        // Apply rotation:
        //   q_rotated = q_even * cos - q_odd * sin
        //   q_rotated = q_even * sin + q_odd * cos (interleaved)
        // Same for k
        ...
    }
}
```

**RoPE scaling for extended context (Llama 3.1 style):**

For models with rope_scaling, the frequency bands are modified:
- **Linear**: `freq /= scaling_factor` — simple interpolation
- **NTK-aware / Llama3**: Different scaling for low vs high frequency bands. Low frequencies (long-range) are interpolated, high frequencies (local) are kept original. Reference the Llama 3.1 paper's approach.

#### SwiGLU Feed-Forward Network

Llama uses SwiGLU instead of standard ReLU FFN. Three weight matrices instead of two:

```
FFN(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
```

Where `silu(x) = x * sigmoid(x)` (also called swish).

```rust
pub struct SwiGLU {
    gate_proj: candle_core::Tensor,  // [intermediate_size, hidden_size]
    up_proj: candle_core::Tensor,    // [intermediate_size, hidden_size]
    down_proj: candle_core::Tensor,  // [hidden_size, intermediate_size]
}

impl SwiGLU {
    pub fn forward(&self, x: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        // 1. gate = x @ gate_proj.T  -> [batch, seq, intermediate_size]
        // 2. up = x @ up_proj.T      -> [batch, seq, intermediate_size]
        // 3. activated = silu(gate) * up
        // 4. output = activated @ down_proj.T -> [batch, seq, hidden_size]
        ...
    }
}
```

#### Grouped-Query Attention (GQA)

Llama 2 70B and all Llama 3 models use GQA, where multiple query heads share a single KV head:

```rust
pub struct LlamaAttention {
    q_proj: candle_core::Tensor,     // [num_heads * head_dim, hidden_size]
    k_proj: candle_core::Tensor,     // [num_kv_heads * head_dim, hidden_size]
    v_proj: candle_core::Tensor,     // [num_kv_heads * head_dim, hidden_size]
    o_proj: candle_core::Tensor,     // [hidden_size, num_heads * head_dim]

    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: Arc<RotaryEmbedding>,
}

impl LlamaAttention {
    pub fn forward(
        &self,
        hidden_states: &candle_core::Tensor,  // [batch, seq_len, hidden_size]
        start_pos: usize,
        kv_cache: &mut KVCache,
        attention_mask: Option<&candle_core::Tensor>,
    ) -> Result<candle_core::Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // 1. Project Q, K, V
        // q: [batch, seq_len, num_heads * head_dim]
        // k: [batch, seq_len, num_kv_heads * head_dim]
        // v: [batch, seq_len, num_kv_heads * head_dim]

        // 2. Reshape for multi-head
        // q: [batch, seq_len, num_heads, head_dim]
        // k: [batch, seq_len, num_kv_heads, head_dim]
        // v: [batch, seq_len, num_kv_heads, head_dim]

        // 3. Apply RoPE to Q and K
        // let (q, k) = self.rope.apply(&q, &k, start_pos)?;

        // 4. Update KV cache (append new K, V)
        // let (k, v) = kv_cache.update(layer_idx, k, v)?;

        // 5. Expand KV heads if using GQA
        // If num_kv_heads < num_heads, repeat K and V:
        //   k: [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        //   Each KV head is repeated (num_heads / num_kv_heads) times

        // 6. Transpose for attention: [batch, num_heads, seq_len, head_dim]

        // 7. Compute attention scores: Q @ K^T / sqrt(head_dim)

        // 8. Apply causal mask (if provided)
        //    For prefill: full causal mask
        //    For decode: no mask needed (single token attends to all past)

        // 9. Softmax

        // 10. Attention output: scores @ V

        // 11. Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]

        // 12. Output projection
        ...
    }
}
```

**GQA head expansion** — this is the key operation. When `num_kv_heads=8` and `num_heads=32`, each KV head serves 4 query heads. Expand by repeating along the head dimension:

```rust
/// Expand KV heads for GQA
/// Input:  [batch, num_kv_heads, seq_len, head_dim]
/// Output: [batch, num_heads, seq_len, head_dim]
fn expand_kv_heads(
    kv: &candle_core::Tensor,
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<candle_core::Tensor> {
    if num_heads == num_kv_heads {
        return Ok(kv.clone()); // MHA, no expansion needed
    }
    let repeat_factor = num_heads / num_kv_heads;
    // Use repeat_interleave or manual reshape+expand
    // kv: [batch, num_kv_heads, seq, head_dim]
    //  -> [batch, num_kv_heads, 1, seq, head_dim]
    //  -> [batch, num_kv_heads, repeat_factor, seq, head_dim]
    //  -> [batch, num_heads, seq, head_dim]
    ...
}
```

### 2.4 Transformer Block (`model/block.rs`)

A single transformer layer. Llama uses pre-norm (norm before attention/FFN, not after):

```rust
pub struct TransformerBlock {
    attention_norm: RMSNorm,
    attention: LlamaAttention,
    ffn_norm: RMSNorm,
    ffn: SwiGLU,
    layer_idx: usize,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        hidden_states: &candle_core::Tensor,
        start_pos: usize,
        kv_cache: &mut KVCache,
        attention_mask: Option<&candle_core::Tensor>,
    ) -> Result<candle_core::Tensor> {
        // Pre-norm architecture (different from original transformer):

        // 1. Attention sub-layer with residual
        let residual = hidden_states;
        let hidden_states = self.attention_norm.forward(hidden_states)?;
        let hidden_states = self.attention.forward(&hidden_states, start_pos, kv_cache, attention_mask)?;
        let hidden_states = (residual + &hidden_states)?; // residual connection

        // 2. FFN sub-layer with residual
        let residual = &hidden_states;
        let hidden_states = self.ffn_norm.forward(&hidden_states)?;
        let hidden_states = self.ffn.forward(&hidden_states)?;
        let hidden_states = (residual + &hidden_states)?; // residual connection

        Ok(hidden_states)
    }
}
```

### 2.5 Full Model (`model/llama.rs`)

```rust
pub struct LlamaModel {
    config: LlamaConfig,
    token_embedding: candle_core::Tensor,  // [vocab_size, hidden_size]
    layers: Vec<TransformerBlock>,
    output_norm: RMSNorm,
    lm_head: candle_core::Tensor,          // [vocab_size, hidden_size]
    device: candle_core::Device,
}

impl LlamaModel {
    /// Load a Llama model from a GGUF file
    pub fn from_gguf<P: AsRef<Path>>(
        path: P,
        device: &candle_core::Device,
    ) -> Result<(Self, Tokenizer)> {
        let mapped = MappedGGUF::open(path)?;
        let config = LlamaConfig::from_gguf(&mapped.gguf)?;

        info!(
            "Loading Llama model: {} layers, {} heads ({} KV), dim {}, vocab {}",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_kv_heads,
            config.hidden_size,
            config.vocab_size,
        );

        // Load embedding
        let token_embedding = mapped.load_tensor(TensorNames::TOKEN_EMBD, device)?;

        // Load layers
        let rope = Arc::new(RotaryEmbedding::new(&config, device)?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = TransformerBlock::load(&mapped, &config, i, rope.clone(), device)?;
            layers.push(block);
            debug!("Loaded layer {}/{}", i + 1, config.num_hidden_layers);
        }

        // Load output
        let output_norm = RMSNorm::new(
            mapped.load_tensor(TensorNames::OUTPUT_NORM, device)?,
            config.rms_norm_eps,
        );

        // LM head — some models tie embeddings (output.weight == token_embd.weight)
        let lm_head = if mapped.has_tensor(TensorNames::OUTPUT) {
            mapped.load_tensor(TensorNames::OUTPUT, device)?
        } else {
            token_embedding.clone() // weight tying
        };

        // Load tokenizer from GGUF metadata
        let tokenizer = Tokenizer::from_gguf(&mapped.gguf)?;

        Ok((Self { config, token_embedding, layers, output_norm, lm_head, device: device.clone() }, tokenizer))
    }

    /// Forward pass: tokens -> logits
    /// input_ids: [batch, seq_len] (u32 token IDs)
    /// start_pos: position offset (0 for prefill, incremented for generation)
    pub fn forward(
        &self,
        input_ids: &candle_core::Tensor,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<candle_core::Tensor> {
        // 1. Token embedding lookup
        let mut hidden_states = self.token_embedding.embedding(input_ids)?;

        // 2. Build causal attention mask (only needed during prefill with seq_len > 1)
        let seq_len = input_ids.dims()[1];
        let mask = if seq_len > 1 {
            Some(self.build_causal_mask(seq_len, start_pos)?)
        } else {
            None // Single token decode step — no mask needed
        };

        // 3. Pass through all transformer blocks
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, start_pos, kv_cache, mask.as_ref())?;
        }

        // 4. Final norm
        let hidden_states = self.output_norm.forward(&hidden_states)?;

        // 5. LM head projection -> logits
        // Only compute logits for the last token (during generation)
        let last_hidden = if seq_len > 1 {
            hidden_states.narrow(1, seq_len - 1, 1)? // [batch, 1, hidden]
        } else {
            hidden_states
        };

        let logits = last_hidden.matmul(&self.lm_head.t()?)?;
        // logits: [batch, 1, vocab_size]

        Ok(logits.squeeze(1)?) // [batch, vocab_size]
    }

    /// Build causal attention mask
    /// Returns mask of shape [1, 1, seq_len, seq_len + start_pos]
    /// Upper triangle = -inf, lower triangle + diagonal = 0
    fn build_causal_mask(&self, seq_len: usize, start_pos: usize) -> Result<candle_core::Tensor> {
        let total_len = start_pos + seq_len;
        // Create [seq_len, total_len] mask where mask[i][j] = 0 if j <= i + start_pos, else -inf
        ...
    }

    pub fn config(&self) -> &LlamaConfig { &self.config }
}
```

---

## Part 3: KV Cache

### 3.1 KV Cache Implementation (`cache/kv.rs`)

The KV cache stores past key/value tensors so they don't need to be recomputed during autoregressive generation:

```rust
pub struct KVCache {
    /// Per-layer cache: (key_cache, value_cache)
    layers: Vec<LayerCache>,
    /// Current sequence length (number of tokens cached)
    current_len: usize,
    /// Maximum cache size
    max_len: usize,
}

struct LayerCache {
    /// Cached keys: [batch, num_kv_heads, cached_len, head_dim]
    key: Option<candle_core::Tensor>,
    /// Cached values: [batch, num_kv_heads, cached_len, head_dim]
    value: Option<candle_core::Tensor>,
}

impl KVCache {
    /// Create a new empty KV cache
    pub fn new(num_layers: usize, max_len: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerCache { key: None, value: None }).collect(),
            current_len: 0,
            max_len,
        }
    }

    /// Update cache for a specific layer, appending new K/V
    /// new_k, new_v: [batch, num_kv_heads, new_seq_len, head_dim]
    /// Returns full (k, v) including cached values
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: candle_core::Tensor,
        new_v: candle_core::Tensor,
    ) -> Result<(candle_core::Tensor, candle_core::Tensor)> {
        let cache = &mut self.layers[layer_idx];

        let (full_k, full_v) = match (&cache.key, &cache.value) {
            (Some(cached_k), Some(cached_v)) => {
                // Concatenate along sequence dimension (dim=2)
                let k = candle_core::Tensor::cat(&[cached_k, &new_k], 2)?;
                let v = candle_core::Tensor::cat(&[cached_v, &new_v], 2)?;
                (k, v)
            }
            _ => (new_k, new_v), // First token, no cache yet
        };

        cache.key = Some(full_k.clone());
        cache.value = Some(full_v.clone());

        // Track sequence length (use layer 0 as reference)
        if layer_idx == 0 {
            self.current_len = full_k.dims()[2];
        }

        Ok((full_k, full_v))
    }

    /// Reset the cache (for new generation)
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.key = None;
            layer.value = None;
        }
        self.current_len = 0;
    }

    /// Current number of cached tokens
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.layers.iter().map(|l| {
            let k_size = l.key.as_ref().map_or(0, |t| t.elem_count() * 4); // assume f32
            let v_size = l.value.as_ref().map_or(0, |t| t.elem_count() * 4);
            k_size + v_size
        }).sum()
    }
}
```

**Memory budget awareness**: For a 7B model with f16 KV cache at 4096 context:
- Per layer: 2 (K+V) × 32 heads × 4096 seq × 128 dim × 2 bytes = 64 MB
- 32 layers: 2 GB total KV cache

For 128K context models, this explodes to ~64 GB. The KV cache compression feature we designed earlier directly addresses this.

---

## Part 4: Tokenizer

### 4.1 BPE Tokenizer from GGUF (`tokenizer/bpe.rs`)

GGUF files embed the tokenizer vocabulary and merge rules in metadata:

```rust
pub struct Tokenizer {
    /// Token ID -> token string/bytes
    vocab: Vec<TokenEntry>,
    /// Token string -> token ID (for encoding)
    token_to_id: HashMap<Vec<u8>, u32>,
    /// BPE merge rules, ordered by priority
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Special token IDs
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: Option<u32>,
    /// Token type flags
    token_types: Vec<TokenType>,
}

pub struct TokenEntry {
    pub text: Vec<u8>,   // Raw bytes (may not be valid UTF-8 for byte tokens)
    pub score: f32,      // Merge priority score
}

pub enum TokenType {
    Normal,
    Unknown,
    Control,    // BOS, EOS, etc.
    UserDefined,
    Unused,
    Byte,       // Byte-level fallback tokens (<0x00> through <0xFF>)
}

impl Tokenizer {
    /// Load tokenizer from GGUF metadata
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self> {
        // Read "tokenizer.ggml.tokens" -> array of strings (the vocabulary)
        // Read "tokenizer.ggml.scores" -> array of f32 (merge priorities)
        // Read "tokenizer.ggml.token_type" -> array of i32 (token types)
        // Read "tokenizer.ggml.merges" -> array of strings (optional, for BPE)
        // Read "tokenizer.ggml.bos_token_id" -> u32
        // Read "tokenizer.ggml.eos_token_id" -> u32
        //
        // The tokenizer type is in "tokenizer.ggml.model":
        //   "llama" -> SentencePiece BPE (Llama 1/2 style)
        //   "gpt2"  -> GPT-2 BPE (Llama 3 style)
        ...
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let mut tokens = if add_bos {
            vec![self.bos_token_id]
        } else {
            vec![]
        };

        // BPE encoding algorithm:
        // 1. Convert text to initial byte/character tokens
        // 2. Iteratively merge the highest-priority adjacent pair
        // 3. Repeat until no more merges possible
        //
        // For SentencePiece-style (Llama 1/2):
        //   - Prepend space (▁) to text unless first token
        //   - Use unigram/BPE hybrid
        //
        // For GPT-2 style (Llama 3):
        //   - Apply regex-based pre-tokenization
        //   - Standard BPE within each word
        ...

        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &token_id in tokens {
            if token_id == self.bos_token_id || token_id == self.eos_token_id {
                continue; // Skip special tokens in output
            }
            if let Some(entry) = self.vocab.get(token_id as usize) {
                // Handle byte tokens: <0xNN> -> actual byte
                if matches!(self.token_types.get(token_id as usize), Some(TokenType::Byte)) {
                    if let Some(byte_val) = Self::parse_byte_token(&entry.text) {
                        bytes.push(byte_val);
                        continue;
                    }
                }
                bytes.extend_from_slice(&entry.text);
            }
        }
        // Convert bytes to string, handling the SentencePiece space marker (▁ -> space)
        let text = String::from_utf8_lossy(&bytes)
            .replace('▁', " ");
        Ok(text)
    }

    /// Decode a single token to its string representation (for streaming)
    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.decode(&[token_id])
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }
    pub fn bos_token_id(&self) -> u32 { self.bos_token_id }
    pub fn eos_token_id(&self) -> u32 { self.eos_token_id }
}
```

**Implementation note on BPE**: The BPE encoding can be implemented using a priority queue. For each adjacent pair, compute its merge priority from the vocabulary scores, and greedily merge the highest-priority pair. Repeat until no mergeable pairs remain. This is well-documented in the SentencePiece and HuggingFace tokenizers codebases.

For MVP, a straightforward O(n²) BPE implementation is fine — tokenization is not the bottleneck. Optimize later if profiling shows it matters.

---

## Part 5: Text Generation

### 5.1 Sampling Strategies (`generate/sampler.rs`)

```rust
pub struct SamplingConfig {
    pub temperature: f32,        // 0.0 = greedy, 0.6-1.0 typical
    pub top_p: f32,              // Nucleus sampling threshold (0.9 typical)
    pub top_k: usize,            // Top-K filtering (40 typical, 0 = disabled)
    pub repeat_penalty: f32,     // Penalize repeated tokens (1.1 typical)
    pub repeat_window: usize,    // How far back to check for repeats (64 typical)
    pub seed: Option<u64>,       // RNG seed for reproducibility
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            repeat_window: 64,
            seed: None,
        }
    }
}

pub struct Sampler {
    config: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Sample a token from logits
    /// logits: [vocab_size] raw model output
    /// past_tokens: recently generated tokens (for repetition penalty)
    pub fn sample(
        &mut self,
        logits: &candle_core::Tensor,
        past_tokens: &[u32],
    ) -> Result<u32> {
        let mut logits_vec = logits.to_vec1::<f32>()?;

        // 1. Apply repetition penalty
        self.apply_repeat_penalty(&mut logits_vec, past_tokens);

        // 2. Temperature scaling
        if self.config.temperature == 0.0 {
            // Greedy: return argmax
            return Ok(logits_vec.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0 as u32);
        }
        for logit in &mut logits_vec {
            *logit /= self.config.temperature;
        }

        // 3. Top-K filtering
        if self.config.top_k > 0 {
            self.apply_top_k(&mut logits_vec);
        }

        // 4. Top-P (nucleus) filtering
        if self.config.top_p < 1.0 {
            self.apply_top_p(&mut logits_vec);
        }

        // 5. Softmax to get probabilities
        let probs = Self::softmax(&logits_vec);

        // 6. Sample from distribution
        self.sample_from_distribution(&probs)
    }

    fn apply_repeat_penalty(&self, logits: &mut [f32], past_tokens: &[u32]) {
        let window_start = past_tokens.len().saturating_sub(self.config.repeat_window);
        for &token in &past_tokens[window_start..] {
            let logit = &mut logits[token as usize];
            if *logit > 0.0 {
                *logit /= self.config.repeat_penalty;
            } else {
                *logit *= self.config.repeat_penalty;
            }
        }
    }

    fn apply_top_k(&self, logits: &mut [f32]) {
        // Find the k-th largest value, set everything below it to -inf
        ...
    }

    fn apply_top_p(&self, logits: &mut [f32]) {
        // Sort by probability, accumulate until sum >= top_p,
        // set remaining to -inf
        ...
    }
}
```

### 5.2 Generation Loop (`generate/stream.rs`)

```rust
pub struct GenerationConfig {
    pub max_new_tokens: usize,     // Maximum tokens to generate
    pub stop_tokens: Vec<u32>,     // Tokens that end generation (EOS, etc.)
    pub sampling: SamplingConfig,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            stop_tokens: vec![],  // Will be populated from model's EOS
            sampling: SamplingConfig::default(),
        }
    }
}

/// Token-by-token generation result
pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    pub text: String,
    pub stats: GenerationStats,
}

pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_time_ms: u64,       // Time for prefill
    pub generation_time_ms: u64,   // Time for all decode steps
    pub tokens_per_second: f64,    // Decode throughput
}

/// Generate text from a prompt
pub fn generate(
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<GenerationOutput> {
    let mut sampler = Sampler::new(config.sampling.clone());
    let mut kv_cache = KVCache::new(
        model.config().num_hidden_layers,
        model.config().max_position_embeddings,
    );

    // Encode prompt
    let prompt_tokens = tokenizer.encode(prompt, true)?; // add BOS
    let prompt_len = prompt_tokens.len();

    let mut all_tokens = prompt_tokens.clone();
    let mut generated_tokens = Vec::new();

    // === Prefill phase ===
    // Process entire prompt in one forward pass
    let prefill_start = std::time::Instant::now();

    let input_ids = candle_core::Tensor::new(
        &prompt_tokens[..],
        model.device(),
    )?.unsqueeze(0)?; // [1, prompt_len]

    let logits = model.forward(&input_ids, 0, &mut kv_cache)?;
    let prefill_time = prefill_start.elapsed();

    // Sample first token from prompt logits
    let next_token = sampler.sample(&logits.squeeze(0)?, &all_tokens)?;
    all_tokens.push(next_token);
    generated_tokens.push(next_token);

    // === Decode phase ===
    // Generate one token at a time
    let decode_start = std::time::Instant::now();

    for _ in 1..config.max_new_tokens {
        // Check stop condition
        if config.stop_tokens.contains(generated_tokens.last().unwrap()) {
            break;
        }

        // Forward pass with single token
        let input_ids = candle_core::Tensor::new(
            &[*generated_tokens.last().unwrap()],
            model.device(),
        )?.unsqueeze(0)?; // [1, 1]

        let start_pos = all_tokens.len() - 1;
        let logits = model.forward(&input_ids, start_pos, &mut kv_cache)?;

        // Sample next token
        let next_token = sampler.sample(&logits.squeeze(0)?, &all_tokens)?;
        all_tokens.push(next_token);
        generated_tokens.push(next_token);
    }

    let decode_time = decode_start.elapsed();
    let gen_count = generated_tokens.len();

    // Decode output
    let output_text = tokenizer.decode(&generated_tokens)?;

    Ok(GenerationOutput {
        tokens: generated_tokens,
        text: output_text,
        stats: GenerationStats {
            prompt_tokens: prompt_len,
            generated_tokens: gen_count,
            prompt_time_ms: prefill_time.as_millis() as u64,
            generation_time_ms: decode_time.as_millis() as u64,
            tokens_per_second: gen_count as f64 / decode_time.as_secs_f64(),
        },
    })
}

/// Streaming generation — yields tokens as they're generated
pub fn generate_stream(
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerationConfig,
    callback: impl FnMut(StreamEvent) -> bool, // return false to stop
) -> Result<GenerationStats> {
    // Same as generate() but calls callback after each token
    // StreamEvent::Token { token_id, text, is_eos }
    // StreamEvent::Done { stats }
    ...
}

pub enum StreamEvent {
    Token {
        token_id: u32,
        text: String,
        is_eos: bool,
    },
    Done {
        stats: GenerationStats,
    },
}
```

---

## Part 6: Public API (`lib.rs`)

```rust
//! RONN-Llama: Native Llama Model Support
//!
//! Load and run Llama models from GGUF files with efficient
//! autoregressive text generation.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use ronn_llama::{LlamaModel, GenerationConfig, generate};
//!
//! // Load model (includes tokenizer)
//! let device = candle_core::Device::Cpu;
//! let (model, tokenizer) = LlamaModel::from_gguf("model.gguf", &device)?;
//!
//! // Generate text
//! let config = GenerationConfig {
//!     max_new_tokens: 128,
//!     stop_tokens: vec![tokenizer.eos_token_id()],
//!     ..Default::default()
//! };
//!
//! let output = generate(&model, &tokenizer, "Once upon a time", &config)?;
//! println!("{}", output.text);
//! println!("{:.1} tokens/sec", output.stats.tokens_per_second);
//! ```

pub mod cache;
pub mod error;
pub mod generate;
pub mod gguf;
pub mod model;
pub mod tokenizer;

pub use cache::kv::KVCache;
pub use error::{LlamaError, Result};
pub use generate::sampler::{Sampler, SamplingConfig};
pub use generate::stream::{
    GenerationConfig, GenerationOutput, GenerationStats, StreamEvent,
    generate, generate_stream,
};
pub use gguf::mmap::MappedGGUF;
pub use gguf::parser::GGUFFile;
pub use gguf::types::GGMLType;
pub use model::config::LlamaConfig;
pub use model::llama::LlamaModel;
pub use tokenizer::bpe::Tokenizer;
```

---

## Part 7: Error Types (`error.rs`)

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LlamaError {
    #[error("GGUF parse error: {0}")]
    GGUFParse(String),

    #[error("Invalid GGUF magic number: expected 0x46475547, got 0x{0:08X}")]
    InvalidMagic(u32),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Missing required metadata key: {0}")]
    MissingMetadata(String),

    #[error("Unsupported quantization type: {0:?}")]
    UnsupportedQuantization(u32),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, LlamaError>;
```

---

## Part 8: Example Binary

Create `examples/llama-generate/`:

```rust
//! Example: Load a GGUF model and generate text
//!
//! Usage:
//!   cargo run --example llama-generate -- --model path/to/model.gguf --prompt "Hello"

use clap::Parser;
use ronn_llama::prelude::*;

#[derive(Parser)]
struct Args {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: String,

    /// Prompt text
    #[arg(short, long, default_value = "Once upon a time")]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    /// Temperature (0.0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-p nucleus sampling
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Use GPU (CUDA)
    #[arg(long)]
    gpu: bool,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::init();

    let args = Args::parse();

    let device = if args.gpu {
        candle_core::Device::new_cuda(0)?
    } else {
        candle_core::Device::Cpu
    };

    println!("Loading model from {}...", args.model);
    let (model, tokenizer) = LlamaModel::from_gguf(&args.model, &device)?;
    println!("Model loaded: {} layers, {} params", ...);

    let config = GenerationConfig {
        max_new_tokens: args.max_tokens,
        stop_tokens: vec![tokenizer.eos_token_id()],
        sampling: SamplingConfig {
            temperature: args.temperature,
            top_p: args.top_p,
            ..Default::default()
        },
    };

    // Stream tokens as they're generated
    print!("{}", args.prompt);
    let stats = generate_stream(&model, &tokenizer, &args.prompt, &config, |event| {
        match event {
            StreamEvent::Token { text, is_eos, .. } => {
                if !is_eos {
                    print!("{}", text);
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
                true // continue
            }
            StreamEvent::Done { stats } => {
                println!("\n\n--- Generation Stats ---");
                println!("Prompt tokens: {}", stats.prompt_tokens);
                println!("Generated tokens: {}", stats.generated_tokens);
                println!("Prefill: {:.1}ms", stats.prompt_time_ms);
                println!("Decode: {:.1} tok/s", stats.tokens_per_second);
                true
            }
        }
    })?;

    Ok(())
}
```

---

## Implementation Order

Build and test in this order, each step producing working code:

### Phase 1: GGUF Parsing (can test with any GGUF file)
1. `gguf/types.rs` — type definitions and constants
2. `gguf/parser.rs` — parse header, metadata, tensor info
3. `gguf/mmap.rs` — memory-mapped file access
4. **Test**: Load a GGUF file, print metadata and tensor names/shapes/types

### Phase 2: Dequantization (test with individual tensors)
5. Implement dequantization for F32, F16, BF16 (trivial)
6. Implement Q8_0 dequantization (simplest quantized format)
7. Implement Q4_0 dequantization
8. Implement Q4_K dequantization (most common)
9. **Test**: Dequantize tensors, verify shapes and value ranges

### Phase 3: Tokenizer (test with text encoding/decoding)
10. `tokenizer/vocab.rs` — load vocabulary from GGUF
11. `tokenizer/bpe.rs` — BPE encode/decode
12. **Test**: Encode "Hello world", decode back, verify roundtrip

### Phase 4: Model Architecture (test layer by layer)
13. `model/config.rs` — extract config from GGUF metadata
14. `model/layers.rs` — RMSNorm (test with known values)
15. `model/layers.rs` — RotaryEmbedding (test with known positions)
16. `model/layers.rs` — SwiGLU FFN
17. `model/layers.rs` — LlamaAttention with GQA
18. `model/block.rs` — TransformerBlock
19. **Test**: Forward pass through single block with random data

### Phase 5: Full Model (test end-to-end)
20. `model/llama.rs` — LlamaModel::from_gguf (load all weights)
21. `cache/kv.rs` — KV cache
22. `model/llama.rs` — forward pass with KV cache
23. **Test**: Load a small GGUF model (e.g., TinyLlama 1.1B Q4_K_M), run forward pass, verify logit shapes

### Phase 6: Generation (test with actual text output)
24. `generate/sampler.rs` — sampling strategies
25. `generate/stream.rs` — generation loop
26. **Test**: Generate text from TinyLlama, verify coherent output

### Phase 7: Polish
27. Error handling cleanup
28. Add `clap` for example binary
29. Performance profiling and obvious optimizations
30. Documentation

---

## Testing Strategy

### Unit Tests
- GGUF parser: test with minimal hand-crafted GGUF binary blobs
- Dequantization: test against known Q4_0/Q8_0 block outputs (values from llama.cpp test suite)
- RMSNorm: test against PyTorch `torch.nn.RMSNorm` output for known inputs
- RoPE: test against known sin/cos rotation values
- BPE: test encode/decode roundtrip with known vocabulary

### Integration Tests
- Download TinyLlama-1.1B-Chat-v1.0 Q4_K_M GGUF (~700MB) as test model
- Verify model loads without error
- Verify tokenizer produces same token IDs as llama.cpp for test strings
- Verify forward pass produces valid logit distributions (sum of softmax ≈ 1.0)
- Verify generation produces coherent text (not just random tokens)

### Performance Benchmarks
- GGUF load time (cold start)
- Prefill throughput (tokens/sec for prompt processing)
- Decode throughput (tokens/sec for generation)
- KV cache memory usage vs. sequence length
- Compare against llama.cpp baseline for same model/hardware

---

## Key Design Decisions

### Why Candle tensors internally (not ronn-core Tensor)?

The existing `ronn_core::Tensor` wraps `candle_core::Tensor` but adds overhead (extra clones, type tracking). For the hot path of transformer inference, we use Candle tensors directly inside the model implementation. The public API can wrap results in `ronn_core::Tensor` if needed for integration.

### Why not use candle-transformers' Llama implementation?

`candle-transformers` is already a workspace dependency and has a Llama implementation. However:
1. It doesn't load GGUF files (only safetensors/HuggingFace format)
2. It doesn't integrate with RONN's provider system
3. We need full control of the forward pass for optimization features (early exit, speculative decoding, etc.)
4. We can reference candle-transformers' implementation as a correctness check

### Weight quantization during inference

For MVP, we dequantize weights to f32 at load time. This is simple and correct but uses more memory. Future optimization:
- Keep weights in quantized format, dequantize on-the-fly during matmul
- This is what llama.cpp does and it's critical for fitting large models in memory
- Connects directly to the BitNet provider work — BitNet's XNOR matmul IS quantized-weight inference

### Thread safety

The model struct is read-only after loading — it can be `Arc<LlamaModel>` shared across threads. The KV cache is per-generation-session and not shared. This naturally supports the continuous batching feature.

---

## Future Integration Points

Once this crate works, the optimization features connect directly:

| Feature | Integration Point |
|---------|------------------|
| **Speculative Decoding** | Use small Llama (1B) as draft model, large Llama (70B) as target. Both loaded via `ronn-llama`. |
| **KV Cache Compression** | Modify `KVCache` to use paged attention, quantized storage, and tiered eviction |
| **Early Exit** | Add confidence checking after each `TransformerBlock`, skip remaining layers if confident |
| **Activation Sparsity** | Profile neuron activations in `SwiGLU`, skip near-zero neurons |
| **Continuous Batching** | Multiple `KVCache` instances, batched `forward()` with different sequence lengths |
| **HRM System 1/2** | Route based on per-token entropy: low entropy → fewer layers (System 1), high entropy → full model (System 2) |
