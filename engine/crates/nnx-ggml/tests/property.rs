use nnx_ggml::{GGMLArchHint, GGMLFile};
use proptest::prelude::*;
use std::io::Write;
use std::panic;
use std::time::{SystemTime, UNIX_EPOCH};

fn arch_hint() -> GGMLArchHint {
    GGMLArchHint {
        architecture: "llama".into(),
        num_layers: 1,
        hidden_dim: 2,
        num_heads: 1,
        vocab_size: 8,
    }
}

fn temp_path() -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("nnx_ggml_prop_{}.bin", stamp))
}

proptest! {
    #[test]
    fn ggml_header_fuzz_does_not_panic(trailing in prop::collection::vec(any::<u8>(), 0..128)) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&nnx_ggml::loader::GGML_MAGIC_V2.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&trailing);

        let path = temp_path();
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();

        let result = panic::catch_unwind(|| GGMLFile::open(&path, arch_hint()));
        std::fs::remove_file(path).ok();
        prop_assert!(result.is_ok());
    }
}