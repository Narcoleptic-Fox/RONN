use nnx_quant::GGMLType;
use proptest::prelude::*;

fn valid_ggml_values() -> Vec<u32> {
    vec![
        0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30,
    ]
}

proptest! {
    #[test]
    fn ggml_type_roundtrip_and_block_metadata_are_consistent(raw in prop::sample::select(valid_ggml_values())) {
        let dtype = GGMLType::from_u32(raw).expect("selected value must map to a GGML type");
        prop_assert_eq!(dtype as u32, raw);
        prop_assert!(!dtype.name().is_empty());
        prop_assert!(dtype.block_numel() > 0);
        if !dtype.is_quantized() {
            prop_assert!(dtype.block_size_bytes() > 0);
        }
    }
}