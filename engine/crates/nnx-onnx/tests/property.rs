use nnx_onnx::OnnxParser;
use proptest::prelude::*;
use std::panic;

proptest! {
    #[test]
    fn arbitrary_payloads_do_not_panic_during_parse(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let result = panic::catch_unwind(|| OnnxParser::parse_bytes(&bytes));
        prop_assert!(result.is_ok());
    }
}