use nnx_safetensors::SafeTensorsFile;
use proptest::prelude::*;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path() -> std::path::PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("nnx_safetensors_prop_{}.safetensors", stamp))
}

fn build_safetensors_bytes(name: &str, values: &[f32]) -> Vec<u8> {
    let data: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    let header = serde_json::json!({
        name: {
            "dtype": "F32",
            "shape": [values.len()],
            "data_offsets": [0, data.len()],
        }
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&data);
    bytes
}

proptest! {
    #[test]
    fn valid_single_tensor_headers_roundtrip(name in "[A-Za-z_][A-Za-z0-9_]{0,15}", values in prop::collection::vec(-10.0f32..10.0, 1..16)) {
        let path = temp_path();
        let bytes = build_safetensors_bytes(&name, &values);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();

        let parsed = SafeTensorsFile::open(&path).unwrap();
        let view = parsed.tensor_view(&name).unwrap();
        prop_assert_eq!(parsed.tensor_names(), vec![name.as_str()]);
        prop_assert_eq!(view.shape().dims(), &[values.len()]);
        prop_assert_eq!(view.as_bytes().len(), values.len() * 4);

        std::fs::remove_file(path).ok();
    }
}