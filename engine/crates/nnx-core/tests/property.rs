use nnx_core::Shape;
use proptest::prelude::*;

proptest! {
    #[test]
    fn shape_numel_matches_dimension_product(dims in prop::collection::vec(1usize..8, 0..4)) {
        let shape = Shape::from(dims.clone());
        let expected = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };

        prop_assert_eq!(shape.dims(), dims.as_slice());
        prop_assert_eq!(shape.numel(), expected);
    }
}