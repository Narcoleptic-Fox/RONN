use nnx_onnx::{OnnxParser, Result};
use std::collections::BTreeMap;

/// Build a minimal synthetic ONNX model protobuf for testing GPT-2 architecture loading.
fn build_synthetic_gpt2_model() -> Vec<u8> {
    use prost::Message;

    let model = nnx_onnx::proto::ModelProto {
        ir_version: 7,
        producer_name: "test".to_string(),
        graph: Some(nnx_onnx::proto::GraphProto {
            name: "gpt2_test".to_string(),
            initializer: vec![
                // Token embeddings: [vocab_size=50, hidden_dim=64]
                make_tensor("transformer.wte.weight", vec![50, 64], 50 * 64),
                // Final layer norm
                make_tensor("transformer.ln_f.weight", vec![64], 64),
                make_tensor("transformer.ln_f.bias", vec![64], 64),
                // LM head
                make_tensor("lm_head.weight", vec![50, 64], 50 * 64),
                // Layer 0
                make_tensor("transformer.h.0.ln_1.weight", vec![64], 64),
                make_tensor("transformer.h.0.ln_1.bias", vec![64], 64),
                make_tensor("transformer.h.0.attn.c_attn.weight", vec![192, 64], 192 * 64), // Fused Q/K/V
                make_tensor("transformer.h.0.attn.c_attn.bias", vec![192], 192),
                make_tensor("transformer.h.0.attn.c_proj.weight", vec![64, 64], 64 * 64),
                make_tensor("transformer.h.0.attn.c_proj.bias", vec![64], 64),
                make_tensor("transformer.h.0.ln_2.weight", vec![64], 64),
                make_tensor("transformer.h.0.ln_2.bias", vec![64], 64),
                make_tensor("transformer.h.0.mlp.c_fc.weight", vec![256, 64], 256 * 64),
                make_tensor("transformer.h.0.mlp.c_fc.bias", vec![256], 256),
                make_tensor("transformer.h.0.mlp.c_proj.weight", vec![64, 256], 64 * 256),
                make_tensor("transformer.h.0.mlp.c_proj.bias", vec![64], 64),
            ],
            ..Default::default()
        }),
        metadata_props: vec![
            make_metadata("hidden_size", "64"),
            make_metadata("vocab_size", "50"),
            make_metadata("num_hidden_layers", "1"),
            make_metadata("num_attention_heads", "4"),
            make_metadata("intermediate_size", "256"),
            make_metadata("max_position_embeddings", "1024"),
            make_metadata("model_type", "gpt2"),
        ],
        ..Default::default()
    };

    let mut buf = Vec::new();
    model.encode(&mut buf).unwrap();
    buf
}

fn make_tensor(name: &str, dims: Vec<i64>, elem_count: usize) -> nnx_onnx::proto::TensorProto {
    use nnx_onnx::proto::tensor_proto::DataType;

    nnx_onnx::proto::TensorProto {
        name: name.to_string(),
        dims,
        data_type: DataType::Float as i32,
        float_data: vec![0.1f32; elem_count],
        ..Default::default()
    }
}

fn make_metadata(key: &str, value: &str) -> nnx_onnx::proto::StringStringEntryProto {
    nnx_onnx::proto::StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    }
}

#[test]
fn test_load_gpt2_architecture() -> Result<()> {
    let bytes = build_synthetic_gpt2_model();
    let parsed = OnnxParser::parse_bytes(&bytes)?;

    // Test architecture-aware loading with GPT-2 family
    let model = parsed.load_dense_model("GPT2LMHeadModel")?;

    // Verify model was loaded with correct architecture
    assert!(
        matches!(
            model.config.arch,
            nnx_transformer::config::Architecture::GPT2
        ),
        "Expected GPT2 architecture, got {:?}",
        model.config.arch
    );

    // Verify config fields match our synthetic model
    assert_eq!(model.config.vocab_size, 50);
    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.num_layers, 1);
    assert_eq!(model.config.num_heads, 4);
    assert_eq!(model.config.head_dim, 64 / 4); // 16
    assert_eq!(model.config.intermediate_dim, 256);
    assert_eq!(model.config.max_context_length, 1024);

    Ok(())
}

#[test]
fn test_load_llama_architecture() -> Result<()> {
    // Build minimal Llama-style model
    use prost::Message;

    let model_proto = nnx_onnx::proto::ModelProto {
        ir_version: 7,
        producer_name: "test".to_string(),
        graph: Some(nnx_onnx::proto::GraphProto {
            name: "llama_test".to_string(),
            initializer: vec![
                make_tensor("model.embed_tokens.weight", vec![100, 128], 100 * 128),
                make_tensor("model.norm.weight", vec![128], 128),
                make_tensor("lm_head.weight", vec![100, 128], 100 * 128),
                make_tensor("model.layers.0.input_layernorm.weight", vec![128], 128),
                make_tensor("model.layers.0.self_attn.q_proj.weight", vec![128, 128], 128 * 128),
                make_tensor("model.layers.0.self_attn.k_proj.weight", vec![128, 128], 128 * 128),
                make_tensor("model.layers.0.self_attn.v_proj.weight", vec![128, 128], 128 * 128),
                make_tensor("model.layers.0.self_attn.o_proj.weight", vec![128, 128], 128 * 128),
                make_tensor("model.layers.0.post_attention_layernorm.weight", vec![128], 128),
                make_tensor("model.layers.0.mlp.gate_proj.weight", vec![512, 128], 512 * 128),
                make_tensor("model.layers.0.mlp.up_proj.weight", vec![512, 128], 512 * 128),
                make_tensor("model.layers.0.mlp.down_proj.weight", vec![128, 512], 128 * 512),
            ],
            ..Default::default()
        }),
        metadata_props: vec![
            make_metadata("hidden_size", "128"),
            make_metadata("vocab_size", "100"),
            make_metadata("num_hidden_layers", "1"),
            make_metadata("num_attention_heads", "8"),
            make_metadata("num_key_value_heads", "8"),
            make_metadata("intermediate_size", "512"),
            make_metadata("model_type", "llama"),
        ],
        ..Default::default()
    };

    let mut bytes = Vec::new();
    model_proto.encode(&mut bytes).unwrap();

    let parsed = OnnxParser::parse_bytes(&bytes)?;
    let model = parsed.load_dense_model("LlamaForCausalLM")?;

    assert!(
        matches!(
            model.config.arch,
            nnx_transformer::config::Architecture::Llama
        ),
        "Expected Llama architecture"
    );
    assert_eq!(model.config.vocab_size, 100);
    assert_eq!(model.config.hidden_dim, 128);
    assert_eq!(model.config.num_layers, 1);

    Ok(())
}

#[test]
fn test_parser_load_model_convenience() -> Result<()> {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let bytes = build_synthetic_gpt2_model();
    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(&bytes).unwrap();
    temp_file.flush().unwrap();

    // Test the OnnxParser::load_model convenience method
    let model = OnnxParser::load_model(temp_file.path(), "GPT2LMHeadModel")?;

    assert!(
        matches!(
            model.config.arch,
            nnx_transformer::config::Architecture::GPT2
        ),
        "Expected GPT2 architecture from convenience method"
    );

    Ok(())
}

#[test]
fn test_unsupported_architecture() {
    let bytes = build_synthetic_gpt2_model();
    let parsed = OnnxParser::parse_bytes(&bytes).unwrap();

    let result = parsed.load_dense_model("UnsupportedArchitecture");
    assert!(
        result.is_err(),
        "Should fail with unrecognized architecture"
    );

    let err_msg = match result {
        Err(err) => format!("{err:?}"),
        Ok(_) => panic!("expected unsupported architecture to fail"),
    };
    assert!(
        err_msg.contains("Unrecognized") || err_msg.contains("UnsupportedArchitecture"),
        "Error should mention unrecognized architecture: {}",
        err_msg
    );
}
