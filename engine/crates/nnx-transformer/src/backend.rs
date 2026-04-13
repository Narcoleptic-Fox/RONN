//! `InferenceEngine` trait implementation — the contract between RONN and NNX.
//!
//! This module wraps the entire NNX transformer pipeline behind the
//! `InferenceEngine` trait defined in `nnx-core`. RONN programs against
//! that trait and never touches NNX internals directly.

use crate::config::ModelConfig;
use crate::generate::{self, GenerateConfig, StopReason};
use crate::loader;
use crate::model::Model;
use crate::sampler::SamplerConfig;
use crate::tokenizer::Tokenizer;
use nnx_core::device::Device;
use nnx_core::engine::*;
use nnx_core::error::{EngineError, Result};
use nnx_core::shape::Shape;
use nnx_core::tensor::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::info;

/// NNX backend implementing `InferenceEngine`.
///
/// Manages loaded models and exposes them through the trait interface.
pub struct NnxBackend {
    models: Mutex<HashMap<u64, LoadedModel>>,
    next_handle: Mutex<u64>,
}

struct LoadedModel {
    model: Model,
    tokenizer: Option<Tokenizer>,
    info: ModelInfo,
    /// Which layers to extract features from on the next forward pass.
    feature_layers: Vec<usize>,
}

impl NnxBackend {
    pub fn new() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(1),
        }
    }

    fn next_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        let id = *h;
        *h += 1;
        id
    }
}

impl Default for NnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngine for NnxBackend {
    fn load_model(&self, path: &Path, config: &LoadConfig) -> Result<ModelHandle> {
        info!("NnxBackend: loading model from {}", path.display());

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let mut model = match ext {
            "gguf" => loader::load_gguf(path)
                .map_err(|e| EngineError::ModelLoad(e))?,
            _ => return Err(EngineError::UnsupportedFormat(
                format!("unsupported file extension: .{} (expected .gguf)", ext)
            )),
        };

        // Override context length if requested
        if config.context_length > 0 {
            model.config.max_context_length = config.context_length;
            // Rebuild cache with new size
            model.cache = crate::cache::KVCache::new(
                model.config.num_layers,
                config.context_length,
                model.config.num_kv_heads,
                model.config.head_dim,
            );
        }

        // Try to load tokenizer from GGUF metadata
        let tokenizer = if ext == "gguf" {
            let gguf = nnx_gguf::GGUFFile::open(path)
                .map_err(|e| EngineError::ModelLoad(format!("failed to reopen GGUF for tokenizer: {}", e)))?;
            Tokenizer::from_gguf(&gguf.metadata).ok()
        } else {
            None
        };

        let cfg = &model.config;
        let info = ModelInfo {
            architecture: cfg.architecture.clone(),
            num_layers: cfg.num_layers,
            hidden_dim: cfg.hidden_dim,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            intermediate_dim: cfg.intermediate_dim,
            vocab_size: cfg.vocab_size,
            max_context_length: cfg.max_context_length,
            num_parameters: cfg.estimated_params(),
            file_size_bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            quantization: "dequantized-f32".into(), // TODO: track original quant type
        };

        info!(
            "Model loaded: {} — {}L/{}H/{} — {:.1}B params",
            info.architecture, info.num_layers, info.num_heads,
            info.hidden_dim, info.num_parameters as f64 / 1e9
        );

        let handle_id = self.next_handle();
        let loaded = LoadedModel {
            model,
            tokenizer,
            info,
            feature_layers: Vec::new(),
        };

        self.models.lock().unwrap().insert(handle_id, loaded);
        Ok(ModelHandle(handle_id))
    }

    fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        self.models
            .lock()
            .unwrap()
            .remove(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        info!("Model unloaded: handle {}", handle.0);
        Ok(())
    }

    fn forward(&self, handle: ModelHandle, input: &TokenBatch) -> Result<GenerationOutput> {
        let mut models = self.models.lock().unwrap();
        let loaded = models.get_mut(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;

        // Process each sequence in the batch (currently sequential)
        // For the common single-sequence case this is fine.
        let mut all_logits = Vec::new();

        for seq in &input.token_ids {
            let mut last_logits = Vec::new();
            for &token in seq {
                last_logits = loaded.model.forward_token(token);
            }
            all_logits.extend_from_slice(&last_logits);
        }

        let vocab_size = loaded.model.config.vocab_size;
        let batch_size = input.batch_size();

        let logits_tensor = Tensor::from_f32(
            &all_logits,
            Shape::new(&[batch_size, vocab_size]),
        )?;

        Ok(GenerationOutput {
            logits: logits_tensor,
            layer_features: None, // TODO: extract when feature_layers is set
        })
    }

    fn forward_layers(
        &self,
        _handle: ModelHandle,
        _input: &Tensor,
        _start_layer: usize,
        _end_layer: usize,
    ) -> Result<Tensor> {
        // TODO: implement partial forward pass for early-exit and self-speculative
        Err(EngineError::Kernel("forward_layers not yet implemented".into()))
    }

    fn model_info(&self, handle: ModelHandle) -> Result<&ModelInfo> {
        // NOTE: This returns a reference into the mutex-guarded map.
        // In practice, RONN should call this once and cache the result.
        // For now, this won't compile as-is because of the borrow checker.
        // We'll need to restructure for production use.
        // Workaround: use a separate Arc<ModelInfo> stored outside the mutex.
        Err(EngineError::ModelLoad("use model_info_cloned() instead".into()))
    }

    fn kv_cache(&self, _handle: ModelHandle) -> Result<&mut dyn KVCacheAccess> {
        // Same borrow issue as model_info — needs architectural revision
        // for the mutex-based approach. For now, callers use reset_cache().
        Err(EngineError::Cache("direct kv_cache access not available through mutex; use dedicated methods".into()))
    }

    fn request_layer_features(&self, handle: ModelHandle, layer_indices: &[usize]) -> Result<()> {
        let mut models = self.models.lock().unwrap();
        let loaded = models.get_mut(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        loaded.feature_layers = layer_indices.to_vec();
        Ok(())
    }
}

/// Extension methods beyond the trait — NNX-specific convenience APIs.
impl NnxBackend {
    /// Get a clone of the model info (avoids borrow issues with the trait method).
    pub fn model_info_cloned(&self, handle: ModelHandle) -> Result<ModelInfo> {
        let models = self.models.lock().unwrap();
        let loaded = models.get(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        Ok(loaded.info.clone())
    }

    /// Reset KV cache for a model (start new conversation).
    pub fn reset_cache(&self, handle: ModelHandle) -> Result<()> {
        let mut models = self.models.lock().unwrap();
        let loaded = models.get_mut(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        loaded.model.reset();
        Ok(())
    }

    /// Current sequence position in the KV cache.
    pub fn cache_position(&self, handle: ModelHandle) -> Result<usize> {
        let models = self.models.lock().unwrap();
        let loaded = models.get(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        Ok(loaded.model.position())
    }

    /// Generate text from a prompt string. Returns generated text.
    ///
    /// This is the high-level "just run it" API.
    pub fn generate_text(
        &self,
        handle: ModelHandle,
        prompt: &str,
        config: &GenerateConfig,
    ) -> Result<String> {
        let mut models = self.models.lock().unwrap();
        let loaded = models.get_mut(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;

        let tokenizer = loaded.tokenizer.as_ref()
            .ok_or_else(|| EngineError::Generation("no tokenizer loaded".into()))?;

        // Encode prompt
        let mut prompt_tokens = vec![tokenizer.bos_token_id];
        prompt_tokens.extend(tokenizer.encode(prompt));

        // Reset cache for new generation
        loaded.model.reset();

        // Generate
        let gen_config = GenerateConfig {
            eos_token_id: tokenizer.eos_token_id,
            ..config.clone()
        };

        let output = generate::generate(&mut loaded.model, &prompt_tokens, &gen_config);

        // Decode output tokens
        let text = tokenizer.decode(&output.tokens);
        Ok(text)
    }

    /// Get the tokenizer for a loaded model (if available).
    pub fn tokenizer(&self, handle: ModelHandle) -> Result<Option<Tokenizer>> {
        let models = self.models.lock().unwrap();
        let loaded = models.get(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        Ok(loaded.tokenizer.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockWeights;
    use crate::model::ModelWeights;

    /// Create a backend with a tiny model loaded (no GGUF file needed).
    fn backend_with_tiny_model() -> (NnxBackend, ModelHandle) {
        let backend = NnxBackend::new();

        let config = ModelConfig {
            architecture: "test".into(),
            num_layers: 1,
            hidden_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_dim: 16,
            vocab_size: 32,
            max_context_length: 64,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
        };

        let hd = 8;
        let weights = ModelWeights {
            token_embedding: vec![0.1; 32 * hd],
            layers: vec![BlockWeights {
                attn_norm: vec![1.0; hd],
                ffn_norm: vec![1.0; hd],
                wq: vec![0.01; 8 * hd],
                wk: vec![0.01; 8 * hd],
                wv: vec![0.01; 8 * hd],
                wo: vec![0.01; hd * 8],
                w_gate: vec![0.01; 16 * hd],
                w_up: vec![0.01; 16 * hd],
                w_down: vec![0.01; hd * 16],
            }],
            final_norm: vec![1.0; hd],
            lm_head: vec![0.01; 32 * hd],
        };

        let model = Model::new(config.clone(), weights);
        let info = ModelInfo {
            architecture: config.architecture.clone(),
            num_layers: config.num_layers,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_dim: config.intermediate_dim,
            vocab_size: config.vocab_size,
            max_context_length: config.max_context_length,
            num_parameters: config.estimated_params(),
            file_size_bytes: 0,
            quantization: "f32".into(),
        };

        let handle_id = backend.next_handle();
        backend.models.lock().unwrap().insert(handle_id, LoadedModel {
            model,
            tokenizer: None,
            info,
            feature_layers: Vec::new(),
        });

        (backend, ModelHandle(handle_id))
    }

    #[test]
    fn test_backend_forward() {
        let (backend, handle) = backend_with_tiny_model();

        let input = TokenBatch::single(vec![1, 2, 3], 0);
        let output = backend.forward(handle, &input).unwrap();

        assert_eq!(output.logits.shape().dims(), &[1, 32]);
    }

    #[test]
    fn test_backend_model_info() {
        let (backend, handle) = backend_with_tiny_model();
        let info = backend.model_info_cloned(handle).unwrap();

        assert_eq!(info.architecture, "test");
        assert_eq!(info.num_layers, 1);
        assert_eq!(info.vocab_size, 32);
    }

    #[test]
    fn test_backend_cache_reset() {
        let (backend, handle) = backend_with_tiny_model();

        let input = TokenBatch::single(vec![1, 2], 0);
        backend.forward(handle, &input).unwrap();
        assert_eq!(backend.cache_position(handle).unwrap(), 2);

        backend.reset_cache(handle).unwrap();
        assert_eq!(backend.cache_position(handle).unwrap(), 0);
    }

    #[test]
    fn test_backend_unload() {
        let (backend, handle) = backend_with_tiny_model();

        backend.unload_model(handle).unwrap();

        // Should fail now
        let result = backend.model_info_cloned(handle);
        assert!(result.is_err());
    }
}
