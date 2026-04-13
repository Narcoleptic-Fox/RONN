//! Model configuration extracted from GGUF metadata.

use nnx_gguf::GGUFMetadata;

/// All hyperparameters needed to construct a Llama-family model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub max_context_length: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
}

impl ModelConfig {
    /// Extract configuration from GGUF metadata.
    pub fn from_gguf(metadata: &GGUFMetadata) -> Result<Self, String> {
        let architecture = metadata
            .architecture()
            .ok_or("missing general.architecture")?
            .to_string();

        let num_layers = metadata
            .num_layers()
            .ok_or("missing block_count")? as usize;

        let hidden_dim = metadata
            .hidden_dim()
            .ok_or("missing embedding_length")? as usize;

        let num_heads = metadata
            .num_heads()
            .ok_or("missing attention.head_count")? as usize;

        let num_kv_heads = metadata
            .num_kv_heads()
            .unwrap_or(num_heads as u32) as usize;

        let head_dim = hidden_dim / num_heads;

        let intermediate_dim = metadata
            .feed_forward_dim()
            .unwrap_or((hidden_dim * 4 * 2 / 3) as u32) as usize;

        let vocab_size = metadata
            .vocab_size()
            .unwrap_or(32000) as usize;

        let max_context_length = metadata
            .context_length()
            .unwrap_or(4096) as usize;

        let rope_freq_base = metadata
            .rope_freq_base()
            .unwrap_or(10000.0);

        let rms_norm_eps = metadata
            .get(&format!("{architecture}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5);

        Ok(Self {
            architecture,
            num_layers,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size,
            max_context_length,
            rope_freq_base,
            rms_norm_eps,
        })
    }

    /// Total parameter count estimate.
    pub fn estimated_params(&self) -> u64 {
        let embed = self.vocab_size * self.hidden_dim;
        let attn_per_layer = 4 * self.hidden_dim * self.hidden_dim;
        let ffn_per_layer = 3 * self.hidden_dim * self.intermediate_dim;
        let norm_per_layer = 2 * self.hidden_dim;
        let per_layer = attn_per_layer + ffn_per_layer + norm_per_layer;
        let head = self.hidden_dim * self.vocab_size;
        (embed + per_layer * self.num_layers + head) as u64
    }
}
