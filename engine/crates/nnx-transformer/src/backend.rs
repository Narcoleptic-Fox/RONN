//! `InferenceEngine` trait implementation — the contract between RONN and NNX.
//!
//! This module wraps the entire NNX transformer pipeline behind the
//! `InferenceEngine` trait defined in `nnx-core`. RONN programs against
//! that trait and never touches NNX internals directly.

use crate::config::ModelConfig;
use crate::generate::{self, GenerateConfig};
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
use std::sync::Mutex;
use tracing::info;

/// NNX backend implementing `InferenceEngine`.
///
/// Manages loaded models and exposes them through the trait interface.
pub struct NnxBackend {
    models: Mutex<HashMap<u64, LoadedModel>>,
    requests: Mutex<HashMap<u64, LoadedRequest>>,
    next_handle: Mutex<u64>,
    next_request: Mutex<u64>,
}

struct LoadedModel {
    model: Model,
    tokenizer: Option<Tokenizer>,
    info: ModelInfo,
}

struct LoadedRequest {
    model_id: u64,
    cache: crate::cache::KVCache,
    /// Which layers to extract features from on the next forward pass.
    feature_layers: Vec<usize>,
}

impl NnxBackend {
    pub fn new() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
            requests: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(1),
            next_request: Mutex::new(1),
        }
    }

    fn next_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        let id = *h;
        *h += 1;
        id
    }

    fn next_request(&self) -> u64 {
        let mut h = self.next_request.lock().unwrap();
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

fn summarize_gguf_quantization(path: &Path) -> Result<String> {
    let gguf = nnx_gguf::GGUFFile::open(path).map_err(|e| {
        EngineError::ModelLoad(format!(
            "failed to reopen GGUF for quantization summary: {}",
            e
        ))
    })?;

    let mut unique = std::collections::BTreeSet::new();
    for info in gguf.tensors.values() {
        unique.insert(info.dtype.to_string());
    }

    Ok(match unique.len() {
        0 => "unknown".into(),
        1 => unique.into_iter().next().unwrap(),
        _ => "mixed".into(),
    })
}

fn summarize_safetensors_dtype(path: &Path) -> Result<String> {
    let st = nnx_safetensors::SafeTensorsFile::open(path).map_err(|e| {
        EngineError::ModelLoad(format!(
            "failed to reopen SafeTensors for dtype summary: {}",
            e
        ))
    })?;

    let mut unique = std::collections::BTreeSet::new();
    for name in st.tensor_names() {
        if let Some(info) = st.tensor_info(name) {
            unique.insert(format!("{:?}", info.dtype));
        }
    }

    Ok(match unique.len() {
        0 => "unknown".into(),
        1 => unique.into_iter().next().unwrap(),
        _ => "mixed".into(),
    })
}

impl InferenceEngine for NnxBackend {
    fn load_model(&self, path: &Path, config: &LoadConfig) -> Result<ModelHandle> {
        info!("NnxBackend: loading model from {}", path.display());

        if config.device != Device::Cpu {
            return Err(EngineError::Device(format!(
                "device {:?} is not implemented yet; only Cpu is currently supported",
                config.device
            )));
        }
        if config.num_threads > 0 {
            return Err(EngineError::Device(
                "explicit thread control is not implemented yet; use num_threads=0".into(),
            ));
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let mut model = match ext {
            "gguf" => loader::load_gguf_with_budget(path, config.memory_budget)
                .map_err(|e| EngineError::ModelLoad(e))?,
            "safetensors" => loader::load_safetensors_with_budget(path, config.memory_budget)
                .map_err(|e| EngineError::ModelLoad(e))?,
            _ => {
                return Err(EngineError::UnsupportedFormat(format!(
                    "unsupported file extension: .{} (expected .gguf or .safetensors)",
                    ext
                )));
            }
        };

        // Override context length if requested
        if config.context_length > 0 {
            model.config.max_context_length = config.context_length;
        }

        // Try to load tokenizer from GGUF metadata (SafeTensors don't embed tokenizers)
        let tokenizer = match ext {
            "gguf" => {
                let gguf = nnx_gguf::GGUFFile::open(path).map_err(|e| {
                    EngineError::ModelLoad(format!("failed to reopen GGUF for tokenizer: {}", e))
                })?;
                Tokenizer::from_gguf(&gguf.metadata).ok()
            }
            _ => None,
        };

        let quantization = match ext {
            "gguf" => summarize_gguf_quantization(path)?,
            "safetensors" => summarize_safetensors_dtype(path)?,
            _ => "unknown".into(),
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
            quantization,
        };

        info!(
            "Model loaded: {} — {}L/{}H/{} — {:.1}B params",
            info.architecture,
            info.num_layers,
            info.num_heads,
            info.hidden_dim,
            info.num_parameters as f64 / 1e9
        );

        let handle_id = self.next_handle();
        let loaded = LoadedModel {
            model,
            tokenizer,
            info,
        };

        self.models.lock().unwrap().insert(handle_id, loaded);
        Ok(ModelHandle(handle_id))
    }

    fn create_request(&self, handle: ModelHandle) -> Result<RequestHandle> {
        let models = self.models.lock().unwrap();
        let loaded = models
            .get(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;

        let request_id = self.next_request();
        let request = LoadedRequest {
            model_id: handle.0,
            cache: loaded.model.new_cache(),
            feature_layers: Vec::new(),
        };
        drop(models);

        self.requests.lock().unwrap().insert(request_id, request);
        Ok(RequestHandle(request_id))
    }

    fn drop_request(&self, handle: RequestHandle) -> Result<()> {
        self.requests
            .lock()
            .unwrap()
            .remove(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        Ok(())
    }

    fn unload_model(&self, handle: ModelHandle) -> Result<()> {
        self.models
            .lock()
            .unwrap()
            .remove(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;

        self.requests
            .lock()
            .unwrap()
            .retain(|_, request| request.model_id != handle.0);

        info!("Model unloaded: handle {}", handle.0);
        Ok(())
    }

    fn forward(&self, handle: RequestHandle, input: &TokenBatch) -> Result<GenerationOutput> {
        if input.token_ids.len() != input.positions.len() {
            return Err(EngineError::ShapeMismatch(format!(
                "batch size {} does not match positions length {}",
                input.token_ids.len(),
                input.positions.len()
            )));
        }

        let models = self.models.lock().unwrap();
        let mut requests = self.requests.lock().unwrap();
        let request = requests
            .get_mut(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        let loaded = models
            .get(&request.model_id)
            .ok_or_else(|| EngineError::ModelLoad("request refers to unknown model".into()))?;

        let batch_size = input.batch_size();
        let mut all_logits = Vec::new();
        let requested_layers = std::mem::take(&mut request.feature_layers);
        let mut layer_features = None;

        if input.batch_size() <= 1 {
            let seq = input.token_ids.first().map(Vec::as_slice).unwrap_or(&[]);
            let position = *input.positions.first().unwrap_or(&0);
            if position > request.cache.position() {
                return Err(EngineError::Cache(format!(
                    "requested start position {} exceeds cache position {}",
                    position,
                    request.cache.position()
                )));
            }

            request.cache.truncate(position);
            let (last_logits, features) = if seq.len() > 1 {
                loaded.model.forward_batch_with_features(
                    &mut request.cache,
                    seq,
                    &requested_layers,
                )?
            } else if seq.len() == 1 {
                loaded.model.forward_token_with_features(
                    &mut request.cache,
                    seq[0],
                    &requested_layers,
                )?
            } else {
                if requested_layers.is_empty() {
                    (vec![0.0; loaded.model.config.vocab_size], Vec::new())
                } else {
                    return Err(EngineError::Generation(
                        "layer features require at least one token in the sequence".into(),
                    ));
                }
            };
            all_logits.extend_from_slice(&last_logits);
            if !features.is_empty() {
                let tensors = features
                    .into_iter()
                    .map(|(layer_idx, data)| {
                        Tensor::from_f32(&data, Shape::new(&[data.len()]))
                            .map(|tensor| (layer_idx, tensor))
                    })
                    .collect::<Result<Vec<_>>>()?;
                layer_features = Some(tensors);
            }
        } else {
            let base_cache = request.cache.clone();
            let mut feature_map: std::collections::BTreeMap<usize, Vec<f32>> = requested_layers
                .iter()
                .copied()
                .map(|layer| (layer, Vec::new()))
                .collect();
            let hidden_dim = loaded.model.config.hidden_dim;

            for (seq, position) in input.token_ids.iter().zip(input.positions.iter().copied()) {
                if position > base_cache.position() {
                    return Err(EngineError::Cache(format!(
                        "requested start position {} exceeds cache position {}",
                        position,
                        base_cache.position()
                    )));
                }

                let mut branch_cache = base_cache.clone();
                branch_cache.truncate(position);

                let (last_logits, features) = if seq.len() > 1 {
                    loaded.model.forward_batch_with_features(
                        &mut branch_cache,
                        seq,
                        &requested_layers,
                    )?
                } else if seq.len() == 1 {
                    loaded.model.forward_token_with_features(
                        &mut branch_cache,
                        seq[0],
                        &requested_layers,
                    )?
                } else {
                    if requested_layers.is_empty() {
                        (vec![0.0; loaded.model.config.vocab_size], Vec::new())
                    } else {
                        return Err(EngineError::Generation(
                            "batch layer features do not support empty sequences".into(),
                        ));
                    }
                };
                all_logits.extend_from_slice(&last_logits);

                for (layer_idx, data) in features {
                    if let Some(stacked) = feature_map.get_mut(&layer_idx) {
                        stacked.extend_from_slice(&data);
                    }
                }
            }

            if !feature_map.is_empty() {
                let tensors = feature_map
                    .into_iter()
                    .map(|(layer_idx, data)| {
                        if data.len() != batch_size * hidden_dim {
                            return Err(EngineError::ShapeMismatch(format!(
                                "layer {} features had {} values, expected {} for batch {} and hidden_dim {}",
                                layer_idx,
                                data.len(),
                                batch_size * hidden_dim,
                                batch_size,
                                hidden_dim
                            )));
                        }
                        Tensor::from_f32(&data, Shape::new(&[batch_size, hidden_dim]))
                            .map(|tensor| (layer_idx, tensor))
                    })
                    .collect::<Result<Vec<_>>>()?;
                layer_features = Some(tensors);
            }
        }

        let vocab_size = loaded.model.config.vocab_size;

        let logits_tensor = Tensor::from_f32(&all_logits, Shape::new(&[batch_size, vocab_size]))?;

        Ok(GenerationOutput {
            logits: logits_tensor,
            layer_features,
        })
    }

    fn forward_layers(
        &self,
        handle: RequestHandle,
        input: &Tensor,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Tensor> {
        let models = self.models.lock().unwrap();
        let mut requests = self.requests.lock().unwrap();
        let request = requests
            .get_mut(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        let loaded = models
            .get(&request.model_id)
            .ok_or_else(|| EngineError::ModelLoad("request refers to unknown model".into()))?;

        let cfg = &loaded.model.config;

        if start_layer >= cfg.num_layers || end_layer > cfg.num_layers || start_layer >= end_layer {
            return Err(EngineError::Kernel(format!(
                "invalid layer range [{}, {}), model has {} layers",
                start_layer, end_layer, cfg.num_layers
            )));
        }

        if input.dtype() != nnx_core::dtype::DType::F32 {
            return Err(EngineError::DType(format!(
                "forward_layers expects F32 input tensor, got {:?}",
                input.dtype()
            )));
        }

        let hidden_dim = cfg.hidden_dim;
        let input_data = input.as_f32();
        let dims = input.shape().dims();
        let batch_size = match dims {
            [d] if *d == hidden_dim => 1,
            [batch, d] if *d == hidden_dim => *batch,
            _ => 0,
        };
        if batch_size == 0 {
            return Err(EngineError::ShapeMismatch(format!(
                "forward_layers input should be [{}] or [batch, {}], got {}",
                hidden_dim,
                hidden_dim,
                input.shape()
            )));
        }

        let mut hidden = input_data.to_vec();
        for layer_idx in start_layer..end_layer {
            let start_position = request.cache.layer(layer_idx).len();
            if batch_size == 1 {
                crate::block::forward_block(
                    &mut hidden,
                    &loaded.model.weights.layers[layer_idx],
                    request.cache.layer_mut(layer_idx),
                    start_position,
                    cfg,
                )?;
            } else {
                crate::block::forward_block_batch(
                    &mut hidden,
                    batch_size,
                    &loaded.model.weights.layers[layer_idx],
                    request.cache.layer_mut(layer_idx),
                    start_position,
                    cfg,
                )?;
            }
        }

        let shape = if batch_size == 1 {
            Shape::new(&[hidden_dim])
        } else {
            Shape::new(&[batch_size, hidden_dim])
        };
        Tensor::from_f32(&hidden, shape)
    }

    fn model_info(&self, handle: ModelHandle) -> Result<ModelInfo> {
        let models = self.models.lock().unwrap();
        let loaded = models
            .get(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        Ok(loaded.info.clone())
    }

    fn cache_tokens(&self, handle: RequestHandle) -> Result<usize> {
        let requests = self.requests.lock().unwrap();
        let request = requests
            .get(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        Ok(request.cache.position())
    }

    fn cache_capacity(&self, handle: RequestHandle) -> Result<usize> {
        let requests = self.requests.lock().unwrap();
        let request = requests
            .get(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        Ok(request.cache.capacity())
    }

    fn cache_memory_bytes(&self, handle: RequestHandle) -> Result<usize> {
        let requests = self.requests.lock().unwrap();
        let request = requests
            .get(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        Ok(request.cache.memory_bytes())
    }

    fn cache_clear(&self, handle: RequestHandle) -> Result<()> {
        let mut requests = self.requests.lock().unwrap();
        let request = requests
            .get_mut(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        request.cache.clear();
        Ok(())
    }

    fn cache_truncate(&self, handle: RequestHandle, n: usize) -> Result<()> {
        let mut requests = self.requests.lock().unwrap();
        let request = requests
            .get_mut(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        request.cache.truncate(n);
        Ok(())
    }

    fn request_layer_features(&self, handle: RequestHandle, layer_indices: &[usize]) -> Result<()> {
        let models = self.models.lock().unwrap();
        let mut requests = self.requests.lock().unwrap();
        let request = requests
            .get_mut(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        let loaded = models
            .get(&request.model_id)
            .ok_or_else(|| EngineError::ModelLoad("request refers to unknown model".into()))?;

        for &layer_idx in layer_indices {
            if layer_idx >= loaded.model.config.num_layers {
                return Err(EngineError::Kernel(format!(
                    "requested layer {} is out of range for model with {} layers",
                    layer_idx, loaded.model.config.num_layers
                )));
            }
        }

        request.feature_layers = layer_indices.to_vec();
        Ok(())
    }
}

/// Extension methods beyond the trait — NNX-specific convenience APIs.
impl NnxBackend {
    /// Get a clone of the model info (avoids borrow issues with the trait method).
    pub fn model_info_cloned(&self, handle: ModelHandle) -> Result<ModelInfo> {
        let models = self.models.lock().unwrap();
        let loaded = models
            .get(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        Ok(loaded.info.clone())
    }

    /// Reset KV cache for a model (start new conversation).
    pub fn reset_cache(&self, handle: RequestHandle) -> Result<()> {
        let mut requests = self.requests.lock().unwrap();
        let request = requests
            .get_mut(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        request.cache.clear();
        Ok(())
    }

    /// Current sequence position in the KV cache.
    pub fn cache_position(&self, handle: RequestHandle) -> Result<usize> {
        let requests = self.requests.lock().unwrap();
        let request = requests
            .get(&handle.0)
            .ok_or_else(|| EngineError::Cache("request handle not found".into()))?;
        Ok(request.cache.position())
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
        let models = self.models.lock().unwrap();
        let loaded = models
            .get(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;

        let tokenizer = loaded
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Generation("no tokenizer loaded".into()))?;

        // Encode prompt
        let mut prompt_tokens = vec![tokenizer.bos_token_id];
        prompt_tokens.extend(tokenizer.encode(prompt));

        let mut cache = loaded.model.new_cache();

        // Generate
        let gen_config = GenerateConfig {
            eos_token_id: tokenizer.eos_token_id,
            ..config.clone()
        };

        let output = generate::generate(&loaded.model, &mut cache, &prompt_tokens, &gen_config)?;

        // Decode output tokens
        let text = tokenizer.decode(&output.tokens);
        Ok(text)
    }

    /// Get the tokenizer for a loaded model (if available).
    pub fn tokenizer(&self, handle: ModelHandle) -> Result<Option<Tokenizer>> {
        let models = self.models.lock().unwrap();
        let loaded = models
            .get(&handle.0)
            .ok_or_else(|| EngineError::ModelLoad("model handle not found".into()))?;
        Ok(loaded.tokenizer.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockWeights;
    use crate::model::ModelWeights;
    use crate::weights::Matrix;

    /// Create a backend with a tiny model loaded (no GGUF file needed).
    fn backend_with_tiny_model() -> (NnxBackend, ModelHandle, RequestHandle) {
        let backend = NnxBackend::new();

        let mut config = ModelConfig::test_llama(8, 2, 2, 4, 16, 32);
        config.num_layers = 1;

        let hd = 8;
        let weights = ModelWeights {
            token_embedding: Matrix::dense(vec![0.1; 32 * hd], 32, hd),
            layers: vec![BlockWeights::test_no_bias(hd, 2, 2, 4, 16)],
            final_norm: vec![1.0; hd],
            final_norm_bias: None,
            lm_head: Matrix::dense(vec![0.01; 32 * hd], 32, hd),
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
        backend.models.lock().unwrap().insert(
            handle_id,
            LoadedModel {
                model,
                tokenizer: None,
                info,
            },
        );
        let model_handle = ModelHandle(handle_id);
        let request_handle = backend.create_request(model_handle).unwrap();

        (backend, model_handle, request_handle)
    }

    #[test]
    fn test_backend_forward() {
        let (backend, _model, request) = backend_with_tiny_model();

        let input = TokenBatch::single(vec![1, 2, 3], 0);
        let output = backend.forward(request, &input).unwrap();

        assert_eq!(output.logits.shape().dims(), &[1, 32]);
    }

    #[test]
    fn test_backend_model_info() {
        let (backend, model, _request) = backend_with_tiny_model();
        let info = backend.model_info_cloned(model).unwrap();

        assert_eq!(info.architecture, "test");
        assert_eq!(info.num_layers, 1);
        assert_eq!(info.vocab_size, 32);
    }

    #[test]
    fn test_backend_cache_reset() {
        let (backend, _model, request) = backend_with_tiny_model();

        let input = TokenBatch::single(vec![1, 2], 0);
        backend.forward(request, &input).unwrap();
        assert_eq!(backend.cache_position(request).unwrap(), 2);

        backend.reset_cache(request).unwrap();
        assert_eq!(backend.cache_position(request).unwrap(), 0);
    }

    #[test]
    fn test_backend_unload() {
        let (backend, model, request) = backend_with_tiny_model();

        backend.unload_model(model).unwrap();

        // Should fail now
        let result = backend.model_info_cloned(model);
        assert!(result.is_err());
        assert!(backend.cache_position(request).is_err());
    }

    #[test]
    fn test_backend_batch_isolation() {
        let (backend, _model, request) = backend_with_tiny_model();

        backend
            .forward(request, &TokenBatch::single(vec![1, 2], 0))
            .unwrap();
        assert_eq!(backend.cache_position(request).unwrap(), 2);

        let batch = TokenBatch::new(vec![vec![3], vec![4]], vec![0, 2]).unwrap();
        let output = backend.forward(request, &batch).unwrap();
        assert_eq!(output.logits.shape().dims(), &[2, 32]);
        assert_eq!(backend.cache_position(request).unwrap(), 2);
    }

    #[test]
    fn test_backend_layer_features() {
        let (backend, _model, request) = backend_with_tiny_model();
        backend.request_layer_features(request, &[0]).unwrap();

        let output = backend
            .forward(request, &TokenBatch::single(vec![1], 0))
            .unwrap();
        let features = output.layer_features.expect("missing layer features");
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].0, 0);
        assert_eq!(features[0].1.shape().dims(), &[8]);
    }

    #[test]
    fn test_backend_layer_features_are_one_shot() {
        let (backend, _model, request) = backend_with_tiny_model();
        backend.request_layer_features(request, &[0]).unwrap();

        let first = backend
            .forward(request, &TokenBatch::single(vec![1], 0))
            .unwrap();
        assert!(first.layer_features.is_some());

        let second = backend
            .forward(request, &TokenBatch::single(vec![2], 1))
            .unwrap();
        assert!(second.layer_features.is_none());
    }

    #[test]
    fn test_backend_layer_features_clear_after_error() {
        let (backend, _model, request) = backend_with_tiny_model();
        backend.request_layer_features(request, &[0]).unwrap();

        let invalid_batch = TokenBatch::new(vec![vec![], vec![1]], vec![0, 0]).unwrap();
        assert!(backend.forward(request, &invalid_batch).is_err());

        let next = backend
            .forward(request, &TokenBatch::single(vec![2], 0))
            .unwrap();
        assert!(next.layer_features.is_none());
    }

    #[test]
    fn test_backend_batch_layer_features_shape() {
        let (backend, _model, request) = backend_with_tiny_model();
        backend.request_layer_features(request, &[0]).unwrap();

        let batch = TokenBatch::new(vec![vec![1], vec![2]], vec![0, 0]).unwrap();
        let output = backend.forward(request, &batch).unwrap();
        let features = output.layer_features.expect("missing layer features");
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].0, 0);
        assert_eq!(features[0].1.shape().dims(), &[2, 8]);
    }

    #[test]
    fn test_backend_forward_layers_batch() {
        let (backend, _model, request) = backend_with_tiny_model();
        let input = Tensor::from_f32(&vec![0.25; 16], Shape::new(&[2, 8])).unwrap();

        let output = backend.forward_layers(request, &input, 0, 1).unwrap();
        assert_eq!(output.shape().dims(), &[2, 8]);
        assert_eq!(backend.cache_position(request).unwrap(), 2);
    }

    #[test]
    fn test_request_layer_features_rejects_out_of_range() {
        let (backend, _model, request) = backend_with_tiny_model();
        assert!(backend.request_layer_features(request, &[1]).is_err());
    }

    #[test]
    fn test_rejects_non_cpu_device_before_load() {
        let backend = NnxBackend::new();
        let config = LoadConfig {
            device: Device::Gpu(0),
            ..LoadConfig::default()
        };

        let result = backend.load_model(std::path::Path::new("ignored.gguf"), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_rejects_explicit_thread_count_before_load() {
        let backend = NnxBackend::new();
        let config = LoadConfig {
            num_threads: 4,
            ..LoadConfig::default()
        };

        let result = backend.load_model(std::path::Path::new("ignored.gguf"), &config);
        assert!(result.is_err());
    }
}
