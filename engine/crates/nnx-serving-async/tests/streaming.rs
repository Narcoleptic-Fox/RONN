use futures::future::join_all;
use futures::stream::{SelectAll, StreamExt};
use nnx_serving::backend::ServingEngine;
use nnx_serving::config::ServingConfig;
use nnx_serving_async::{AsyncServingEngine, AsyncServingRequest, SamplerConfig, SequenceId};
use nnx_transformer::block::BlockWeights;
use nnx_transformer::config::*;
use nnx_transformer::model::{Model, ModelWeights};
use nnx_transformer::weights::Matrix;
use tokio::time::{Duration, sleep, timeout};

fn make_model() -> Model {
    let hidden_dim = 16;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let intermediate_dim = 32;
    let vocab_size = 64;
    let num_layers = 2;

    let config = ModelConfig {
        architecture: "test".into(),
        arch: Architecture::Llama,
        num_layers,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_dim,
        vocab_size,
        max_context_length: 256,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        norm_type: NormType::RMSNorm,
        ffn_type: FFNType::SwiGLU,
        pos_encoding: PosEncoding::RoPE { freq_base: 10000.0 },
        block_style: BlockStyle::Sequential,
        has_qkv_bias: false,
        has_output_bias: false,
        embedding_scale: None,
        activation_quantization: ActivationQuantization::None,
    };

    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let make_layer = |seed: usize| BlockWeights {
        attn_norm: vec![1.0; hidden_dim],
        ffn_norm: vec![1.0; hidden_dim],
        wq: Matrix::dense(
            (0..q_dim * hidden_dim)
                .map(|i| ((i + seed) % 7) as f32 * 0.015 - 0.045)
                .collect(),
            q_dim,
            hidden_dim,
        ),
        wk: Matrix::dense(
            (0..kv_dim * hidden_dim)
                .map(|i| ((i + seed) % 11) as f32 * 0.015 - 0.075)
                .collect(),
            kv_dim,
            hidden_dim,
        ),
        wv: Matrix::dense(
            (0..kv_dim * hidden_dim)
                .map(|i| ((i + seed) % 13) as f32 * 0.015 - 0.09)
                .collect(),
            kv_dim,
            hidden_dim,
        ),
        wo: Matrix::dense(
            (0..hidden_dim * q_dim)
                .map(|i| ((i + seed) % 5) as f32 * 0.015 - 0.03)
                .collect(),
            hidden_dim,
            q_dim,
        ),
        w_gate: Matrix::dense(
            vec![0.01; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        ),
        w_up: Matrix::dense(
            vec![0.01; intermediate_dim * hidden_dim],
            intermediate_dim,
            hidden_dim,
        ),
        w_down: Matrix::dense(
            vec![0.01; hidden_dim * intermediate_dim],
            hidden_dim,
            intermediate_dim,
        ),
        bq: None,
        bk: None,
        bv: None,
        bo: None,
        attn_norm_bias: None,
        ffn_norm_bias: None,
    };

    let weights = ModelWeights {
        token_embedding: Matrix::dense(
            (0..vocab_size * hidden_dim)
                .map(|i| (i % 29) as f32 * 0.04 - 0.56)
                .collect(),
            vocab_size,
            hidden_dim,
        ),
        position_embedding: None,
        layers: (0..num_layers).map(|i| make_layer(i * 31)).collect(),
        final_norm: vec![1.0; hidden_dim],
        final_norm_bias: None,
        lm_head: Matrix::dense(
            (0..vocab_size * hidden_dim)
                .map(|i| (i % 37) as f32 * 0.025 - 0.45)
                .collect(),
            vocab_size,
            hidden_dim,
        ),
    };

    Model::new(config, weights)
}

fn serving_config() -> ServingConfig {
    ServingConfig {
        page_size: 4,
        max_pages: 512,
        max_sequences: 8,
        max_batch_size: 8,
        enable_prefix_caching: false,
        max_prefix_cache_entries: 64,
        max_prefill_tokens: 0,
        gpu_kv_quantization: Default::default(),
    }
}

fn request(prompt_tokens: Vec<u32>, max_new_tokens: usize) -> AsyncServingRequest {
    AsyncServingRequest {
        prompt_tokens,
        max_new_tokens,
        sampler: SamplerConfig::greedy(),
        eos_token_id: None,
        seed: 1,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_four_requests_streams_interleaved_tokens() {
    let engine =
        AsyncServingEngine::new(ServingEngine::new(make_model(), serving_config()).unwrap());

    let prompts = [
        vec![1, 2, 3, 4],
        vec![5, 6, 7],
        vec![8, 9],
        vec![10, 11, 12, 13, 14],
    ];

    let handles = join_all(
        prompts
            .into_iter()
            .map(|prompt| engine.submit(request(prompt, 3))),
    )
    .await;

    let mut merged = SelectAll::new();
    let mut ids = Vec::new();
    for handle in handles {
        let handle = handle.unwrap();
        let seq_id = handle.sequence_id();
        ids.push(seq_id);
        let stream = engine
            .stream(seq_id)
            .unwrap()
            .map(move |item| (seq_id, item));
        merged.push(Box::pin(stream));
    }

    let mut first_round = Vec::new();
    while first_round.len() < 4 {
        let (seq_id, output) = timeout(Duration::from_secs(3), merged.next())
            .await
            .expect("timed out waiting for token")
            .expect("merged stream ended early");
        assert_eq!(seq_id, output.seq_id);
        first_round.push(seq_id);
    }

    first_round.sort_by_key(|id| id.0);
    ids.sort_by_key(|id| id.0);
    assert_eq!(
        first_round, ids,
        "first decode round should emit one token per active sequence"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_request_handle_cancels_sequence() {
    let engine =
        AsyncServingEngine::new(ServingEngine::new(make_model(), serving_config()).unwrap());

    let handle = engine.submit(request(vec![1, 2, 3, 4], 16)).await.unwrap();
    let seq_id = handle.sequence_id();
    let mut stream = engine.stream(seq_id).unwrap();

    let _first = timeout(Duration::from_secs(3), stream.next())
        .await
        .expect("timed out waiting for first token")
        .expect("stream ended before first token");

    drop(handle);

    timeout(Duration::from_secs(3), async {
        loop {
            let stats = engine.stats().await.unwrap();
            if stats.running == 0 && stats.waiting == 0 {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("cancellation should be observed promptly");

    timeout(Duration::from_secs(1), async {
        loop {
            match stream.next().await {
                Some(_) => continue,
                None => break,
            }
        }
    })
    .await
    .expect("cancelled sequence stream should close once buffered outputs drain");
}
