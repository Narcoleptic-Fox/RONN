//! Architecture-aware HuggingFace weight name resolution.
//!
//! HuggingFace models use varied naming conventions per architecture family.
//! This module provides a data-table-driven mapping from HF tensor names to
//! NNX internal names, and utilities to:
//!
//! - Detect the layer index from a tensor name
//! - Count transformer layers without architecture-specific knowledge
//! - Split fused QKV tensors (GPT-2, Phi, Qwen-1) into separate Q/K/V
//! - Locate multi-file index files for sharded models

use crate::config::Architecture;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::warn;

// ---------------------------------------------------------------------------
// Name patterns
// ---------------------------------------------------------------------------

/// A single HF-name-to-internal-name mapping entry.
///
/// `hf_pattern` may contain `{i}` which is replaced with the layer index at
/// query time.  `internal` is the NNX internal name, also with `{i}`.
#[derive(Debug, Clone)]
pub struct NameEntry {
    /// The HuggingFace tensor name template (may contain `{i}`).
    pub hf_pattern: &'static str,
    /// The NNX internal name template (may contain `{i}`).
    pub internal: &'static str,
}

impl NameEntry {
    const fn new(hf_pattern: &'static str, internal: &'static str) -> Self {
        Self {
            hf_pattern,
            internal,
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture tables
// ---------------------------------------------------------------------------

/// Non-layer (global) weight mappings for Llama/Mistral.
const LLAMA_GLOBAL: &[NameEntry] = &[
    NameEntry::new("model.embed_tokens.weight", "token_embd.weight"),
    NameEntry::new("model.norm.weight", "output_norm.weight"),
    NameEntry::new("model.norm.bias", "output_norm.bias"),
    NameEntry::new("lm_head.weight", "output.weight"),
];

/// Per-layer weight mappings for Llama/Mistral.  `{i}` is the layer index.
const LLAMA_LAYER: &[NameEntry] = &[
    NameEntry::new(
        "model.layers.{i}.self_attn.q_proj.weight",
        "blk.{i}.attn_q.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.k_proj.weight",
        "blk.{i}.attn_k.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.v_proj.weight",
        "blk.{i}.attn_v.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.o_proj.weight",
        "blk.{i}.attn_output.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.q_proj.bias",
        "blk.{i}.attn_q.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.k_proj.bias",
        "blk.{i}.attn_k.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.v_proj.bias",
        "blk.{i}.attn_v.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.o_proj.bias",
        "blk.{i}.attn_output.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.mlp.gate_proj.weight",
        "blk.{i}.ffn_gate.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.mlp.up_proj.weight",
        "blk.{i}.ffn_up.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.mlp.down_proj.weight",
        "blk.{i}.ffn_down.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.input_layernorm.weight",
        "blk.{i}.attn_norm.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.input_layernorm.bias",
        "blk.{i}.attn_norm.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.post_attention_layernorm.weight",
        "blk.{i}.ffn_norm.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.post_attention_layernorm.bias",
        "blk.{i}.ffn_norm.bias",
    ),
];

/// Gemma global weights (same embedding pattern as Llama, tied lm_head possible).
const GEMMA_GLOBAL: &[NameEntry] = &[
    NameEntry::new("model.embed_tokens.weight", "token_embd.weight"),
    NameEntry::new("model.norm.weight", "output_norm.weight"),
    // Gemma often ties lm_head with embed_tokens; handled separately
    NameEntry::new("lm_head.weight", "output.weight"),
];

/// Gemma per-layer weights are identical to Llama.
const GEMMA_LAYER: &[NameEntry] = LLAMA_LAYER;

/// Qwen2 global weights.  Qwen2 uses the same naming as Llama.
const QWEN_GLOBAL: &[NameEntry] = &[
    NameEntry::new("model.embed_tokens.weight", "token_embd.weight"),
    NameEntry::new("model.norm.weight", "output_norm.weight"),
    NameEntry::new("lm_head.weight", "output.weight"),
];

/// Qwen2 per-layer weights.  Qwen2 uses separate Q/K/V (not fused).
/// Qwen1 uses fused c_attn — detected separately via `FusedQkv`.
const QWEN_LAYER: &[NameEntry] = &[
    NameEntry::new(
        "model.layers.{i}.self_attn.q_proj.weight",
        "blk.{i}.attn_q.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.k_proj.weight",
        "blk.{i}.attn_k.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.v_proj.weight",
        "blk.{i}.attn_v.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.o_proj.weight",
        "blk.{i}.attn_output.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.q_proj.bias",
        "blk.{i}.attn_q.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.k_proj.bias",
        "blk.{i}.attn_k.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.v_proj.bias",
        "blk.{i}.attn_v.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.o_proj.bias",
        "blk.{i}.attn_output.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.mlp.gate_proj.weight",
        "blk.{i}.ffn_gate.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.mlp.up_proj.weight",
        "blk.{i}.ffn_up.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.mlp.down_proj.weight",
        "blk.{i}.ffn_down.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.input_layernorm.weight",
        "blk.{i}.attn_norm.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.post_attention_layernorm.weight",
        "blk.{i}.ffn_norm.weight",
    ),
];

/// Phi global weights.
const PHI_GLOBAL: &[NameEntry] = &[
    NameEntry::new("model.embed_tokens.weight", "token_embd.weight"),
    NameEntry::new("model.final_layernorm.weight", "output_norm.weight"),
    NameEntry::new("model.final_layernorm.bias", "output_norm.bias"),
    NameEntry::new("lm_head.weight", "output.weight"),
    NameEntry::new("lm_head.bias", "output.bias"),
];

/// Phi per-layer weights.  Phi uses a fused QKV projection named `qkv_proj`
/// in some variants, or separate `q_proj`/`k_proj`/`v_proj`.  The fused path
/// is handled separately; these entries cover the separated case.
const PHI_LAYER: &[NameEntry] = &[
    NameEntry::new(
        "model.layers.{i}.self_attn.q_proj.weight",
        "blk.{i}.attn_q.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.k_proj.weight",
        "blk.{i}.attn_k.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.v_proj.weight",
        "blk.{i}.attn_v.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.dense.weight",
        "blk.{i}.attn_output.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.dense.bias",
        "blk.{i}.attn_output.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.q_proj.bias",
        "blk.{i}.attn_q.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.k_proj.bias",
        "blk.{i}.attn_k.bias",
    ),
    NameEntry::new(
        "model.layers.{i}.self_attn.v_proj.bias",
        "blk.{i}.attn_v.bias",
    ),
    NameEntry::new("model.layers.{i}.mlp.fc1.weight", "blk.{i}.ffn_gate.weight"),
    NameEntry::new("model.layers.{i}.mlp.fc1.bias", "blk.{i}.ffn_gate.bias"),
    NameEntry::new("model.layers.{i}.mlp.fc2.weight", "blk.{i}.ffn_down.weight"),
    NameEntry::new("model.layers.{i}.mlp.fc2.bias", "blk.{i}.ffn_down.bias"),
    NameEntry::new(
        "model.layers.{i}.input_layernorm.weight",
        "blk.{i}.attn_norm.weight",
    ),
    NameEntry::new(
        "model.layers.{i}.input_layernorm.bias",
        "blk.{i}.attn_norm.bias",
    ),
];

/// GPT-2 global weights.
const GPT2_GLOBAL: &[NameEntry] = &[
    NameEntry::new("transformer.wte.weight", "token_embd.weight"),
    NameEntry::new("transformer.wpe.weight", "pos_embd.weight"),
    NameEntry::new("transformer.ln_f.weight", "output_norm.weight"),
    NameEntry::new("transformer.ln_f.bias", "output_norm.bias"),
    // GPT-2 ties lm_head with wte; if lm_head.weight is absent we fall back
    NameEntry::new("lm_head.weight", "output.weight"),
];

/// GPT-2 per-layer weights.  The combined QKV `c_attn` tensor is handled via
/// `FusedQkv`; these entries cover what is NOT fused.
const GPT2_LAYER: &[NameEntry] = &[
    NameEntry::new(
        "transformer.h.{i}.attn.c_proj.weight",
        "blk.{i}.attn_output.weight",
    ),
    NameEntry::new(
        "transformer.h.{i}.attn.c_proj.bias",
        "blk.{i}.attn_output.bias",
    ),
    NameEntry::new(
        "transformer.h.{i}.mlp.c_fc.weight",
        "blk.{i}.ffn_gate.weight",
    ),
    NameEntry::new("transformer.h.{i}.mlp.c_fc.bias", "blk.{i}.ffn_gate.bias"),
    NameEntry::new(
        "transformer.h.{i}.mlp.c_proj.weight",
        "blk.{i}.ffn_down.weight",
    ),
    NameEntry::new("transformer.h.{i}.mlp.c_proj.bias", "blk.{i}.ffn_down.bias"),
    NameEntry::new("transformer.h.{i}.ln_1.weight", "blk.{i}.attn_norm.weight"),
    NameEntry::new("transformer.h.{i}.ln_1.bias", "blk.{i}.attn_norm.bias"),
    NameEntry::new("transformer.h.{i}.ln_2.weight", "blk.{i}.ffn_norm.weight"),
    NameEntry::new("transformer.h.{i}.ln_2.bias", "blk.{i}.ffn_norm.bias"),
];

// ---------------------------------------------------------------------------
// Fused QKV description
// ---------------------------------------------------------------------------

/// Describes an architecture that packs Q, K, and V into one tensor.
#[derive(Debug, Clone)]
pub struct FusedQkvDesc {
    /// HF name pattern for the fused weight tensor (may contain `{i}`).
    pub weight_pattern: &'static str,
    /// HF name pattern for the fused bias tensor (may contain `{i}`).
    /// Empty string means no bias.
    pub bias_pattern: &'static str,
}

// ---------------------------------------------------------------------------
// Architecture descriptor
// ---------------------------------------------------------------------------

/// Encapsulates all name-mapping knowledge for a single architecture.
#[derive(Debug)]
pub struct ArchDescriptor {
    pub global_entries: &'static [NameEntry],
    pub layer_entries: &'static [NameEntry],
    /// Some architectures pack Q+K+V into a single tensor that needs splitting.
    pub fused_qkv: Option<FusedQkvDesc>,
    /// The HF name pattern used to count transformer layers (must contain `{i}`).
    pub layer_probe_pattern: &'static str,
}

impl ArchDescriptor {
    fn for_arch(arch: Architecture) -> &'static Self {
        match arch {
            // Llama-family: same HF naming conventions
            Architecture::Llama | Architecture::Mistral | Architecture::CodeLlama => &LLAMA_DESC,
            Architecture::Gemma => &GEMMA_DESC,
            Architecture::Qwen => &QWEN_DESC,
            Architecture::Phi => &PHI_DESC,
            // GPT-2 style (Falcon and MPT also use transformer.h naming)
            Architecture::GPT2 | Architecture::Falcon | Architecture::MPT => &GPT2_DESC,
            // StableLM uses Llama-style naming (model.layers.{i}...)
            Architecture::StableLM => &LLAMA_DESC,
        }
    }
}

static LLAMA_DESC: ArchDescriptor = ArchDescriptor {
    global_entries: LLAMA_GLOBAL,
    layer_entries: LLAMA_LAYER,
    fused_qkv: None,
    layer_probe_pattern: "model.layers.{i}.self_attn.q_proj.weight",
};

static GEMMA_DESC: ArchDescriptor = ArchDescriptor {
    global_entries: GEMMA_GLOBAL,
    layer_entries: GEMMA_LAYER,
    fused_qkv: None,
    layer_probe_pattern: "model.layers.{i}.self_attn.q_proj.weight",
};

static QWEN_DESC: ArchDescriptor = ArchDescriptor {
    global_entries: QWEN_GLOBAL,
    layer_entries: QWEN_LAYER,
    // Qwen2 uses separate projections.  Qwen1 had fused c_attn but we
    // auto-detect that at load time; the layer_probe also works for Qwen1
    // because Qwen1 has `transformer.h.{i}.attn.c_attn.weight` which we
    // detect separately.  For now, the probe below handles Qwen2.
    fused_qkv: None,
    layer_probe_pattern: "model.layers.{i}.self_attn.q_proj.weight",
};

static PHI_DESC: ArchDescriptor = ArchDescriptor {
    global_entries: PHI_GLOBAL,
    layer_entries: PHI_LAYER,
    fused_qkv: Some(FusedQkvDesc {
        weight_pattern: "model.layers.{i}.self_attn.qkv_proj.weight",
        bias_pattern: "model.layers.{i}.self_attn.qkv_proj.bias",
    }),
    layer_probe_pattern: "model.layers.{i}.self_attn.q_proj.weight",
};

static GPT2_DESC: ArchDescriptor = ArchDescriptor {
    global_entries: GPT2_GLOBAL,
    layer_entries: GPT2_LAYER,
    fused_qkv: Some(FusedQkvDesc {
        weight_pattern: "transformer.h.{i}.attn.c_attn.weight",
        bias_pattern: "transformer.h.{i}.attn.c_attn.bias",
    }),
    layer_probe_pattern: "transformer.h.{i}.attn.c_attn.weight",
};

// ---------------------------------------------------------------------------
// WeightNameMap — public API
// ---------------------------------------------------------------------------

/// Resolves HuggingFace tensor names to NNX internal names for a given architecture.
///
/// The resolution is entirely data-table driven: no deeply nested if-else logic.
/// Callers should use `resolve_global` / `resolve_layer` to get HF names for
/// a given internal key, or `parse_hf_name` to go the other direction.
pub struct WeightNameMap {
    arch: Architecture,
    desc: &'static ArchDescriptor,
    /// Maps HF tensor name -> internal name (for global weights, pre-built at construction).
    global_hf_to_internal: HashMap<String, String>,
}

impl WeightNameMap {
    /// Build a resolver for the given architecture.
    pub fn from_architecture(arch: Architecture) -> Self {
        let desc = ArchDescriptor::for_arch(arch.clone());

        let mut global_hf_to_internal = HashMap::new();
        for entry in desc.global_entries {
            global_hf_to_internal.insert(entry.hf_pattern.to_string(), entry.internal.to_string());
        }

        Self {
            arch,
            desc,
            global_hf_to_internal,
        }
    }

    /// Return the HuggingFace tensor name for an internal name at a given layer.
    ///
    /// Returns `None` if the internal name is unknown for this architecture.
    pub fn resolve_layer(&self, internal_suffix: &str, layer: usize) -> Option<String> {
        for entry in self.desc.layer_entries {
            if entry.internal.replace("{i}", &layer.to_string()) == internal_suffix
                || entry.internal == internal_suffix
            {
                return Some(entry.hf_pattern.replace("{i}", &layer.to_string()));
            }
        }
        None
    }

    /// Return the HuggingFace tensor name for a global (non-layer) internal name.
    pub fn resolve_global(&self, internal_name: &str) -> Option<&str> {
        for entry in self.desc.global_entries {
            if entry.internal == internal_name {
                return Some(entry.hf_pattern);
            }
        }
        None
    }

    /// Parse an HF tensor name and return the internal name plus optional layer index.
    ///
    /// For global tensors: `Some((internal_name, None))`
    /// For layer tensors: `Some((internal_name_with_{i}_replaced, Some(layer_idx)))`
    /// For unknown names: `None` (caller may log a warning)
    pub fn parse_hf_name(&self, hf_name: &str) -> Option<(String, Option<usize>)> {
        // Check global table first.
        if let Some(internal) = self.global_hf_to_internal.get(hf_name) {
            return Some((internal.clone(), None));
        }

        // Try layer patterns.
        for entry in self.desc.layer_entries {
            if let Some((layer_idx, internal)) = match_layer_pattern(entry, hf_name) {
                return Some((internal, Some(layer_idx)));
            }
        }

        // Try fused QKV patterns (so callers know the tensor exists but must split it).
        if let Some(fused) = &self.desc.fused_qkv {
            if let Some(layer_idx) = extract_layer_index(fused.weight_pattern, hf_name) {
                let internal = format!("blk.{}.attn_qkv_fused.weight", layer_idx);
                return Some((internal, Some(layer_idx)));
            }
            if !fused.bias_pattern.is_empty() {
                if let Some(layer_idx) = extract_layer_index(fused.bias_pattern, hf_name) {
                    let internal = format!("blk.{}.attn_qkv_fused.bias", layer_idx);
                    return Some((internal, Some(layer_idx)));
                }
            }
        }

        None
    }

    /// Whether this architecture uses fused QKV for the given layer.
    pub fn has_fused_qkv(&self) -> bool {
        self.desc.fused_qkv.is_some()
    }

    /// Return the HF name for the fused QKV weight at the given layer, if applicable.
    pub fn fused_qkv_weight_name(&self, layer: usize) -> Option<String> {
        self.desc
            .fused_qkv
            .as_ref()
            .map(|f| f.weight_pattern.replace("{i}", &layer.to_string()))
    }

    /// Return the HF name for the fused QKV bias at the given layer, if applicable.
    /// Returns `None` if there is no fused QKV bias.
    pub fn fused_qkv_bias_name(&self, layer: usize) -> Option<String> {
        self.desc.fused_qkv.as_ref().and_then(|f| {
            if f.bias_pattern.is_empty() {
                None
            } else {
                Some(f.bias_pattern.replace("{i}", &layer.to_string()))
            }
        })
    }

    /// Count transformer layers by probing the SafeTensors file.
    ///
    /// Uses the architecture-specific probe pattern, so this works for all
    /// supported architectures rather than assuming Llama-style names.
    pub fn count_layers(&self, tensor_names: &dyn Fn(&str) -> bool) -> usize {
        let pattern = self.desc.layer_probe_pattern;

        // For architectures with fused QKV, the probe pattern is the fused tensor.
        // For architectures with split QKV, probe for q_proj or equivalent.
        let mut count = 0;
        loop {
            let name = pattern.replace("{i}", &count.to_string());
            if !tensor_names(&name) {
                break;
            }
            count += 1;
        }
        count
    }

    /// Return all layer-level HF names for a given layer index (excluding fused QKV).
    ///
    /// Useful for enumerating expected tensors when loading.
    pub fn layer_hf_names(&self, layer: usize) -> Vec<String> {
        self.desc
            .layer_entries
            .iter()
            .map(|e| e.hf_pattern.replace("{i}", &layer.to_string()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal pattern matching helpers
// ---------------------------------------------------------------------------

/// Match a layer pattern against an HF name, returning `(layer_idx, internal_name)` on success.
fn match_layer_pattern(entry: &NameEntry, hf_name: &str) -> Option<(usize, String)> {
    let layer_idx = extract_layer_index(entry.hf_pattern, hf_name)?;
    let internal = entry.internal.replace("{i}", &layer_idx.to_string());
    Some((layer_idx, internal))
}

/// Extract the numeric layer index by matching a pattern containing `{i}` against a string.
///
/// Returns `None` if the pattern does not match.
pub fn extract_layer_index(pattern: &str, name: &str) -> Option<usize> {
    let (prefix, suffix) = pattern.split_once("{i}")?;
    let rest = name.strip_prefix(prefix)?;
    let digits_end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits_end == 0 {
        return None;
    }
    let layer_str = &rest[..digits_end];
    let tail = &rest[digits_end..];
    if tail != suffix {
        return None;
    }
    layer_str.parse().ok()
}

// ---------------------------------------------------------------------------
// Multi-file index support
// ---------------------------------------------------------------------------

/// Describes a sharded SafeTensors model split across multiple files.
pub struct ShardIndex {
    /// Maps tensor name -> path of the shard file that contains it.
    pub tensor_to_file: HashMap<String, PathBuf>,
    /// All unique shard file paths, in the order they appear in the index.
    pub shard_files: Vec<PathBuf>,
}

/// Attempt to load a multi-file SafeTensors index from `dir/model.safetensors.index.json`.
///
/// Returns `None` if no index file is present (single-file model).
/// Returns an error if the index file is present but malformed.
pub fn load_shard_index(model_path: &Path) -> Result<Option<ShardIndex>, String> {
    // The index lives next to the .safetensors file, or in the same directory.
    let dir = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path.parent().unwrap_or(Path::new(".")).to_path_buf()
    };

    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(None);
    }

    let text = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("failed to read {}: {}", index_path.display(), e))?;

    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("invalid JSON in {}: {}", index_path.display(), e))?;

    let weight_map = value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("{}: missing or invalid 'weight_map'", index_path.display()))?;

    let mut tensor_to_file: HashMap<String, PathBuf> = HashMap::new();
    let mut seen_files: Vec<PathBuf> = Vec::new();

    for (tensor_name, file_value) in weight_map {
        let file_name = file_value.as_str().ok_or_else(|| {
            format!(
                "{}: weight_map entry for '{}' is not a string",
                index_path.display(),
                tensor_name
            )
        })?;

        let file_path = dir.join(file_name);

        if !seen_files.contains(&file_path) {
            seen_files.push(file_path.clone());
        }
        tensor_to_file.insert(tensor_name.clone(), file_path);
    }

    tracing::info!(
        "Multi-file SafeTensors: {} tensors across {} shard files",
        tensor_to_file.len(),
        seen_files.len()
    );

    Ok(Some(ShardIndex {
        tensor_to_file,
        shard_files: seen_files,
    }))
}

// ---------------------------------------------------------------------------
// QKV splitting
// ---------------------------------------------------------------------------

/// Split a fused [3 * q_dim, hidden_dim] (or [q_dim + 2*kv_dim, hidden_dim]) weight matrix
/// into separate Q, K, V tensors in f32.
///
/// `q_dim = num_heads * head_dim`, `kv_dim = num_kv_heads * head_dim`.
/// For MHA, `q_dim == kv_dim`.  For GQA, `kv_dim < q_dim`.
///
/// The fused layout is assumed to be [Q rows | K rows | V rows] in that order,
/// which is the standard for HuggingFace and PyTorch convention.
pub fn split_fused_qkv_weight(
    fused: &[f32],
    q_dim: usize,
    kv_dim: usize,
    hidden_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let expected = (q_dim + kv_dim + kv_dim) * hidden_dim;
    if fused.len() != expected {
        return Err(format!(
            "fused QKV weight has {} elements but expected {} (q_dim={}, kv_dim={}, hidden_dim={})",
            fused.len(),
            expected,
            q_dim,
            kv_dim,
            hidden_dim,
        ));
    }

    let q_start = 0;
    let k_start = q_dim * hidden_dim;
    let v_start = k_start + kv_dim * hidden_dim;

    let q = fused[q_start..k_start].to_vec();
    let k = fused[k_start..v_start].to_vec();
    let v = fused[v_start..].to_vec();

    Ok((q, k, v))
}

/// Split a fused [q_dim + kv_dim + kv_dim] bias vector into separate Q, K, V biases.
pub fn split_fused_qkv_bias(
    fused: &[f32],
    q_dim: usize,
    kv_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let expected = q_dim + kv_dim + kv_dim;
    if fused.len() != expected {
        return Err(format!(
            "fused QKV bias has {} elements but expected {} (q_dim={}, kv_dim={})",
            fused.len(),
            expected,
            q_dim,
            kv_dim,
        ));
    }

    let k_start = q_dim;
    let v_start = k_start + kv_dim;

    let q = fused[..k_start].to_vec();
    let k = fused[k_start..v_start].to_vec();
    let v = fused[v_start..].to_vec();

    Ok((q, k, v))
}

// ---------------------------------------------------------------------------
// Architecture detection from HF names
// ---------------------------------------------------------------------------

/// Map HuggingFace `architectures` or `model_type` strings to NNX `Architecture`.
///
/// Delegates to the profile registry in `config.rs` when possible, so that
/// both code paths stay in sync.  Falls back to substring matching for names
/// not yet in the registry.
///
/// Returns `None` only for genuinely unknown architectures.
pub fn map_hf_architecture(name: &str) -> Option<Architecture> {
    use crate::config::find_profile_by_hf;

    // Use the canonical profile registry as the source of truth.
    if let Some(profile) = find_profile_by_hf(name) {
        return Some(profile_name_to_arch(profile.name));
    }

    warn!(
        "unrecognized HuggingFace architecture '{}' — not in profile registry",
        name
    );
    None
}

/// Convert a profile name (from `ArchitectureProfile::name`) to the `Architecture` enum.
///
/// This is a thin shim so that `weight_names` does not need to import the whole
/// profile machinery — just the `Architecture` enum.
fn profile_name_to_arch(name: &str) -> Architecture {
    match name {
        "llama" => Architecture::Llama,
        "mistral" => Architecture::Mistral,
        "codellama" => Architecture::CodeLlama,
        "gpt2" => Architecture::GPT2,
        "phi" => Architecture::Phi,
        "gemma" => Architecture::Gemma,
        "qwen" => Architecture::Qwen,
        "stablelm" => Architecture::StableLM,
        "falcon" => Architecture::Falcon,
        "mpt" => Architecture::MPT,
        _ => {
            warn!("unknown profile name '{}', defaulting to Llama", name);
            Architecture::Llama
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // extract_layer_index
    // -------------------------------------------------------------------

    #[test]
    fn test_extract_layer_index_basic() {
        assert_eq!(
            extract_layer_index(
                "model.layers.{i}.self_attn.q_proj.weight",
                "model.layers.3.self_attn.q_proj.weight"
            ),
            Some(3)
        );
        assert_eq!(
            extract_layer_index(
                "model.layers.{i}.self_attn.q_proj.weight",
                "model.layers.0.self_attn.q_proj.weight"
            ),
            Some(0)
        );
        assert_eq!(
            extract_layer_index(
                "model.layers.{i}.self_attn.q_proj.weight",
                "model.layers.31.self_attn.q_proj.weight"
            ),
            Some(31)
        );
    }

    #[test]
    fn test_extract_layer_index_no_match() {
        assert_eq!(
            extract_layer_index(
                "model.layers.{i}.self_attn.q_proj.weight",
                "model.norm.weight"
            ),
            None
        );
        assert_eq!(
            extract_layer_index(
                "model.layers.{i}.self_attn.q_proj.weight",
                "model.layers.abc.self_attn.q_proj.weight"
            ),
            None
        );
        // suffix mismatch
        assert_eq!(
            extract_layer_index(
                "model.layers.{i}.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight"
            ),
            None
        );
    }

    #[test]
    fn test_extract_layer_index_gpt2_pattern() {
        assert_eq!(
            extract_layer_index(
                "transformer.h.{i}.attn.c_attn.weight",
                "transformer.h.0.attn.c_attn.weight"
            ),
            Some(0)
        );
        assert_eq!(
            extract_layer_index(
                "transformer.h.{i}.attn.c_attn.weight",
                "transformer.h.11.attn.c_attn.weight"
            ),
            Some(11)
        );
    }

    // -------------------------------------------------------------------
    // WeightNameMap – Llama
    // -------------------------------------------------------------------

    #[test]
    fn test_llama_global_resolution() {
        let map = WeightNameMap::from_architecture(Architecture::Llama);

        assert_eq!(
            map.resolve_global("token_embd.weight"),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            map.resolve_global("output_norm.weight"),
            Some("model.norm.weight")
        );
        assert_eq!(map.resolve_global("output.weight"), Some("lm_head.weight"));
    }

    #[test]
    fn test_llama_layer_parse() {
        let map = WeightNameMap::from_architecture(Architecture::Llama);

        let result = map.parse_hf_name("model.layers.5.self_attn.q_proj.weight");
        assert_eq!(result, Some(("blk.5.attn_q.weight".to_string(), Some(5))));

        let result = map.parse_hf_name("model.layers.0.mlp.gate_proj.weight");
        assert_eq!(result, Some(("blk.0.ffn_gate.weight".to_string(), Some(0))));

        let result = map.parse_hf_name("model.layers.2.input_layernorm.weight");
        assert_eq!(
            result,
            Some(("blk.2.attn_norm.weight".to_string(), Some(2)))
        );
    }

    #[test]
    fn test_llama_global_parse() {
        let map = WeightNameMap::from_architecture(Architecture::Llama);

        let result = map.parse_hf_name("model.embed_tokens.weight");
        assert_eq!(result, Some(("token_embd.weight".to_string(), None)));

        let result = map.parse_hf_name("lm_head.weight");
        assert_eq!(result, Some(("output.weight".to_string(), None)));
    }

    #[test]
    fn test_llama_unknown_name_returns_none() {
        let map = WeightNameMap::from_architecture(Architecture::Llama);
        assert!(
            map.parse_hf_name("some.completely.unknown.tensor")
                .is_none()
        );
    }

    // -------------------------------------------------------------------
    // WeightNameMap – GPT-2 fused QKV
    // -------------------------------------------------------------------

    #[test]
    fn test_gpt2_has_fused_qkv() {
        let map = WeightNameMap::from_architecture(Architecture::GPT2);
        assert!(map.has_fused_qkv());
        assert_eq!(
            map.fused_qkv_weight_name(0),
            Some("transformer.h.0.attn.c_attn.weight".to_string())
        );
        assert_eq!(
            map.fused_qkv_bias_name(3),
            Some("transformer.h.3.attn.c_attn.bias".to_string())
        );
    }

    #[test]
    fn test_gpt2_fused_qkv_parsed() {
        let map = WeightNameMap::from_architecture(Architecture::GPT2);
        let result = map.parse_hf_name("transformer.h.0.attn.c_attn.weight");
        // Should be recognized as a fused QKV tensor
        assert!(result.is_some());
        let (internal, layer) = result.unwrap();
        assert_eq!(layer, Some(0));
        assert!(internal.contains("attn_qkv_fused"));
    }

    #[test]
    fn test_gpt2_non_fused_tensor_parsed() {
        let map = WeightNameMap::from_architecture(Architecture::GPT2);
        let result = map.parse_hf_name("transformer.h.0.attn.c_proj.weight");
        assert_eq!(
            result,
            Some(("blk.0.attn_output.weight".to_string(), Some(0)))
        );
    }

    #[test]
    fn test_gpt2_layer_count() {
        let map = WeightNameMap::from_architecture(Architecture::GPT2);
        let present: std::collections::HashSet<String> = [
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.1.attn.c_attn.weight",
            "transformer.h.2.attn.c_attn.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let count = map.count_layers(&|name| present.contains(name));
        assert_eq!(count, 3);
    }

    // -------------------------------------------------------------------
    // WeightNameMap – Phi fused QKV
    // -------------------------------------------------------------------

    #[test]
    fn test_phi_has_fused_qkv() {
        let map = WeightNameMap::from_architecture(Architecture::Phi);
        assert!(map.has_fused_qkv());
        assert_eq!(
            map.fused_qkv_weight_name(0),
            Some("model.layers.0.self_attn.qkv_proj.weight".to_string())
        );
    }

    // -------------------------------------------------------------------
    // WeightNameMap – Llama no fused QKV
    // -------------------------------------------------------------------

    #[test]
    fn test_llama_no_fused_qkv() {
        let map = WeightNameMap::from_architecture(Architecture::Llama);
        assert!(!map.has_fused_qkv());
        assert!(map.fused_qkv_weight_name(0).is_none());
        assert!(map.fused_qkv_bias_name(0).is_none());
    }

    // -------------------------------------------------------------------
    // split_fused_qkv_weight
    // -------------------------------------------------------------------

    #[test]
    fn test_split_fused_qkv_weight_mha() {
        // 2 heads, head_dim=4, hidden=8  =>  q_dim=8, kv_dim=8
        // fused shape: [24, 8] = 192 elements
        let q_dim = 8;
        let kv_dim = 8;
        let hidden_dim = 8;
        let total = (q_dim + kv_dim + kv_dim) * hidden_dim;

        // Fill with recognizable values: Q=1.0, K=2.0, V=3.0
        let mut fused = vec![0.0f32; total];
        for i in 0..q_dim * hidden_dim {
            fused[i] = 1.0;
        }
        for i in q_dim * hidden_dim..(q_dim + kv_dim) * hidden_dim {
            fused[i] = 2.0;
        }
        for i in (q_dim + kv_dim) * hidden_dim..total {
            fused[i] = 3.0;
        }

        let (q, k, v) = split_fused_qkv_weight(&fused, q_dim, kv_dim, hidden_dim).unwrap();

        assert_eq!(q.len(), q_dim * hidden_dim);
        assert_eq!(k.len(), kv_dim * hidden_dim);
        assert_eq!(v.len(), kv_dim * hidden_dim);
        assert!(q.iter().all(|&x| x == 1.0));
        assert!(k.iter().all(|&x| x == 2.0));
        assert!(v.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_split_fused_qkv_weight_gqa() {
        // 4 q heads, 2 kv heads, head_dim=4, hidden=16
        // q_dim=16, kv_dim=8
        let q_dim = 16;
        let kv_dim = 8;
        let hidden_dim = 16;
        let total = (q_dim + kv_dim + kv_dim) * hidden_dim;

        let fused = vec![1.0f32; total];
        let (q, k, v) = split_fused_qkv_weight(&fused, q_dim, kv_dim, hidden_dim).unwrap();

        assert_eq!(q.len(), q_dim * hidden_dim);
        assert_eq!(k.len(), kv_dim * hidden_dim);
        assert_eq!(v.len(), kv_dim * hidden_dim);
    }

    #[test]
    fn test_split_fused_qkv_weight_wrong_size() {
        let result = split_fused_qkv_weight(&[1.0; 10], 4, 4, 4);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expected"));
    }

    // -------------------------------------------------------------------
    // split_fused_qkv_bias
    // -------------------------------------------------------------------

    #[test]
    fn test_split_fused_qkv_bias() {
        let q_dim = 8;
        let kv_dim = 4;
        let total = q_dim + kv_dim + kv_dim;
        let mut fused = vec![0.0f32; total];
        for i in 0..q_dim {
            fused[i] = 1.0;
        }
        for i in q_dim..q_dim + kv_dim {
            fused[i] = 2.0;
        }
        for i in q_dim + kv_dim..total {
            fused[i] = 3.0;
        }

        let (bq, bk, bv) = split_fused_qkv_bias(&fused, q_dim, kv_dim).unwrap();

        assert_eq!(bq.len(), q_dim);
        assert_eq!(bk.len(), kv_dim);
        assert_eq!(bv.len(), kv_dim);
        assert!(bq.iter().all(|&x| x == 1.0));
        assert!(bk.iter().all(|&x| x == 2.0));
        assert!(bv.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_split_fused_qkv_bias_wrong_size() {
        let result = split_fused_qkv_bias(&[1.0; 5], 4, 4);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------
    // map_hf_architecture
    // -------------------------------------------------------------------

    #[test]
    fn test_map_hf_architecture_llama_variants() {
        assert_eq!(
            map_hf_architecture("LlamaForCausalLM"),
            Some(Architecture::Llama)
        );
        assert_eq!(
            map_hf_architecture("MistralForCausalLM"),
            Some(Architecture::Mistral)
        );
        assert_eq!(
            map_hf_architecture("CodeLlamaForCausalLM"),
            Some(Architecture::CodeLlama)
        );
        assert_eq!(map_hf_architecture("llama"), Some(Architecture::Llama));
        assert_eq!(map_hf_architecture("mistral"), Some(Architecture::Mistral));
    }

    #[test]
    fn test_map_hf_architecture_gemma() {
        assert_eq!(
            map_hf_architecture("GemmaForCausalLM"),
            Some(Architecture::Gemma)
        );
        assert_eq!(
            map_hf_architecture("Gemma2ForCausalLM"),
            Some(Architecture::Gemma)
        );
    }

    #[test]
    fn test_map_hf_architecture_qwen() {
        assert_eq!(
            map_hf_architecture("Qwen2ForCausalLM"),
            Some(Architecture::Qwen)
        );
        assert_eq!(
            map_hf_architecture("QwenForCausalLM"),
            Some(Architecture::Qwen)
        );
    }

    #[test]
    fn test_map_hf_architecture_phi() {
        assert_eq!(
            map_hf_architecture("PhiForCausalLM"),
            Some(Architecture::Phi)
        );
        assert_eq!(
            map_hf_architecture("Phi3ForCausalLM"),
            Some(Architecture::Phi)
        );
    }

    #[test]
    fn test_map_hf_architecture_gpt2() {
        assert_eq!(
            map_hf_architecture("GPT2LMHeadModel"),
            Some(Architecture::GPT2)
        );
        assert_eq!(map_hf_architecture("gpt2"), Some(Architecture::GPT2));
        assert_eq!(
            map_hf_architecture("GPTJForCausalLM"),
            Some(Architecture::GPT2)
        );
    }

    // -------------------------------------------------------------------
    // ShardIndex
    // -------------------------------------------------------------------

    #[test]
    fn test_load_shard_index_no_file() {
        let temp_dir = std::env::temp_dir().join("nnx_shard_test_no_index");
        std::fs::create_dir_all(&temp_dir).ok();
        let result = load_shard_index(&temp_dir.join("model.safetensors"));
        std::fs::remove_dir_all(&temp_dir).ok();
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_load_shard_index_parses_correctly() {
        use std::io::Write;

        let temp_dir = std::env::temp_dir().join(format!(
            "nnx_shard_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let index = serde_json::json!({
            "metadata": {"total_size": 1234},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors"
            }
        });

        let index_path = temp_dir.join("model.safetensors.index.json");
        let mut f = std::fs::File::create(&index_path).unwrap();
        f.write_all(serde_json::to_string(&index).unwrap().as_bytes())
            .unwrap();
        f.sync_all().unwrap();

        let result = load_shard_index(&temp_dir.join("model.safetensors"));
        std::fs::remove_dir_all(&temp_dir).ok();

        let shard_index = result.unwrap().expect("should find index");
        assert_eq!(shard_index.tensor_to_file.len(), 3);
        assert_eq!(shard_index.shard_files.len(), 2);

        let shard1 = temp_dir.join("model-00001-of-00002.safetensors");
        assert_eq!(
            shard_index.tensor_to_file["model.embed_tokens.weight"],
            shard1
        );
    }
}
