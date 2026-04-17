use nnx_kernels::softmax::softmax_f32;
use proptest::prelude::*;

proptest! {
    #[test]
    fn softmax_outputs_form_a_probability_distribution(values in prop::collection::vec(-20.0f32..20.0, 1..32)) {
        let mut probs = values;
        softmax_f32(&mut probs);

        let sum: f32 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4, "sum was {}", sum);
        for prob in probs {
            prop_assert!(prob.is_finite());
            prop_assert!(prob >= 0.0);
            prop_assert!(prob <= 1.0 + 1e-6);
        }
    }
}