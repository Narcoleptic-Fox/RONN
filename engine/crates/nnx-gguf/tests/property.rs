use nnx_gguf::GGUFFile;
use proptest::prelude::*;
use std::io::Write;
use std::panic;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path() -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("nnx_gguf_prop_{}.gguf", stamp))
}

proptest! {
    #[test]
    fn valid_header_with_arbitrary_trailing_bytes_does_not_panic(trailing in prop::collection::vec(any::<u8>(), 0..128), version in prop_oneof![Just(2u32), Just(3u32)]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x46475547u32.to_le_bytes());
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&trailing);

        let path = temp_path();
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();

        let result = panic::catch_unwind(|| GGUFFile::open(&path));
        std::fs::remove_file(path).ok();
        prop_assert!(result.is_ok());
    }
}