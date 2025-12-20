//! Property-based and fuzzing tests for ONNX parser robustness
//! Tests: Malformed inputs, boundary conditions, invalid protobuf data

use prost::Message;
use ronn_onnx::ModelLoader;
use ronn_onnx::onnx_proto::*;

// Helper functions to create common protobuf structures with less boilerplate
fn dim_value(val: i64) -> tensor_shape_proto::Dimension {
    tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(tensor_shape_proto::dimension::Value::DimValue(val)),
    }
}

fn tensor_type(elem_type: i32, dims: Vec<i64>) -> TypeProto {
    TypeProto {
        denotation: String::new(),
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type,
            shape: Some(TensorShapeProto {
                dim: dims.into_iter().map(dim_value).collect(),
            }),
        })),
    }
}

fn value_info(name: &str, elem_type: i32, dims: Vec<i64>) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(tensor_type(elem_type, dims)),
        ..Default::default()
    }
}

// ============ Malformed Input Tests ============

#[test]
fn test_empty_input() {
    let result = ModelLoader::load_from_bytes(&[]);
    assert!(result.is_err(), "Should reject empty input");
}

#[test]
fn test_random_bytes() {
    let random_data = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA];
    let result = ModelLoader::load_from_bytes(&random_data);
    assert!(result.is_err(), "Should reject random bytes");
}

#[test]
fn test_partial_json() {
    let partial_json = b"{\"ir_version\": 7, \"graph\":";
    let result = ModelLoader::load_from_bytes(partial_json);
    assert!(result.is_err(), "Should reject incomplete JSON");
}

#[test]
fn test_invalid_unicode() {
    let invalid_utf8 = vec![
        b'{', b'"', b'n', b'a', b'm', b'e', b'"', b':', b'"', 0xFF,
        0xFE, // Invalid UTF-8 sequence
        b'"', b'}',
    ];
    let result = ModelLoader::load_from_bytes(&invalid_utf8);
    assert!(result.is_err(), "Should reject invalid UTF-8");
}

// ============ Missing Required Fields ============

#[test]
fn test_missing_graph_name() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "".to_string(), // Empty name
            node: vec![],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should still load - name is optional
    assert!(result.is_ok());
}

#[test]
fn test_missing_node_op_type() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "node1".to_string(),
                op_type: "".to_string(), // Empty op_type
                input: vec![],
                output: vec![],
                attribute: vec![],
                ..Default::default()
            }],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should still load with empty op_type
    assert!(result.is_ok());
}

#[test]
fn test_missing_tensor_dims() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![],
            input: vec![],
            output: vec![],
            initializer: vec![TensorProto {
                name: "weight".to_string(),
                dims: vec![], // Empty dims
                data_type: 1,
                float_data: vec![1.0, 2.0],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // May fail during tensor creation
    // Just ensure it doesn't panic
    let _ = result;
}

// ============ Invalid Type Combinations ============

#[test]
fn test_attribute_type_mismatch() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "test_node".to_string(),
                op_type: "Test".to_string(),
                input: vec![],
                output: vec![],
                attribute: vec![AttributeProto {
                    name: "value".to_string(),
                    r#type: attribute_proto::AttributeType::Float as i32,
                    i: 42, // But providing int value
                    ..Default::default()
                }],
                ..Default::default()
            }],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should handle gracefully (use 0.0 as default)
    assert!(result.is_ok());
}

#[test]
fn test_invalid_data_type() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![],
            input: vec![value_info("input", 999, vec![2])], // Invalid type
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should load but may have issues with type mapping
    // Just ensure no panic
    let _ = result;
}

// ============ Boundary Conditions ============

#[test]
fn test_very_large_tensor_dims() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![],
            input: vec![value_info("huge_input", 1, vec![1000000, 1000000])],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should be able to parse the shape without allocating tensor
    assert!(result.is_ok());
}

#[test]
fn test_zero_dimensions() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![],
            input: vec![value_info("zero_dim", 1, vec![0])],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.inputs[0].shape, vec![0]);
}

#[test]
fn test_negative_dimensions() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![],
            input: vec![value_info("negative_dim", 1, vec![-1])],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Negative dimensions might be interpreted as large positive numbers
    // Should not panic
    let _ = result;
}

// ============ Deep Nesting ============

#[test]
fn test_deeply_nested_graph() {
    // Create a model with many nodes
    let mut nodes = Vec::new();
    for i in 0..100 {
        nodes.push(NodeProto {
            name: format!("node_{}", i),
            op_type: "Relu".to_string(),
            input: vec![format!("input_{}", i)],
            output: vec![format!("output_{}", i)],
            attribute: vec![],
            ..Default::default()
        });
    }

    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "deep_graph".to_string(),
            node: nodes,
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle many nodes");

    let model = result.unwrap();
    let node_count = model.graph.nodes();
    assert_eq!(node_count.len(), 100);
}

// ============ String Handling ============

#[test]
fn test_very_long_names() {
    let long_name = "a".repeat(10000);

    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: long_name,
            node: vec![],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle long names");
}

#[test]
fn test_empty_string_names() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "".to_string(),
            node: vec![NodeProto {
                name: "".to_string(),
                op_type: "".to_string(),
                input: vec!["".to_string()],
                output: vec!["".to_string()],
                attribute: vec![],
                ..Default::default()
            }],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle empty strings");
}

#[test]
fn test_special_characters_in_names() {
    let special_chars = "!@#$%^&*()[]{}|\\:;'\"<>,.?/~`";

    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: special_chars.to_string(),
            node: vec![],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle special characters");
}

// ============ Array Handling ============

#[test]
fn test_empty_arrays() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "test_node".to_string(),
                op_type: "Test".to_string(),
                input: vec![],  // Empty input array
                output: vec![], // Empty output array
                attribute: vec![AttributeProto {
                    name: "empty_ints".to_string(),
                    ints: vec![],
                    r#type: attribute_proto::AttributeType::Ints as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());
}

#[test]
fn test_very_large_arrays() {
    let large_array: Vec<i64> = (0..10000).collect();

    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "test_node".to_string(),
                op_type: "Test".to_string(),
                input: vec![],
                output: vec![],
                attribute: vec![AttributeProto {
                    name: "large_ints".to_string(),
                    ints: large_array,
                    r#type: attribute_proto::AttributeType::Ints as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle large arrays");
}

// ============ Circular Reference Prevention ============

#[test]
fn test_node_output_as_input() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "cycle".to_string(),
                op_type: "Add".to_string(),
                input: vec!["x".to_string(), "y".to_string()],
                output: vec!["x".to_string()], // Output same as input
                attribute: vec![],
                ..Default::default()
            }],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Should parse but graph validation might catch issues
    assert!(result.is_ok());
}

// ============ Numeric Edge Cases ============

#[test]
fn test_extreme_float_values() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "extreme".to_string(),
                op_type: "Test".to_string(),
                input: vec![],
                output: vec![],
                attribute: vec![
                    AttributeProto {
                        name: "infinity".to_string(),
                        f: f32::INFINITY,
                        r#type: attribute_proto::AttributeType::Float as i32,
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "neg_infinity".to_string(),
                        f: f32::NEG_INFINITY,
                        r#type: attribute_proto::AttributeType::Float as i32,
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nan".to_string(),
                        f: f32::NAN,
                        r#type: attribute_proto::AttributeType::Float as i32,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    // Protobuf should support infinity/nan
    let _ = result;
}

#[test]
fn test_extreme_int_values() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "extreme".to_string(),
                op_type: "Test".to_string(),
                input: vec![],
                output: vec![],
                attribute: vec![
                    AttributeProto {
                        name: "max_i64".to_string(),
                        i: i64::MAX,
                        r#type: attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "min_i64".to_string(),
                        i: i64::MIN,
                        r#type: attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok(), "Should handle extreme integers");
}

// ============ Memory Safety ============

#[test]
fn test_large_file_size() {
    // Create a model with reasonable content but test parsing doesn't allocate excessively
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "memory_test".to_string(),
            node: vec![],
            input: vec![],
            output: vec![],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut model_bytes = model_proto.encode_to_vec();

    // Pad with extra bytes (will be ignored by protobuf parser)
    model_bytes.extend(vec![0; 1000]);

    let result = ModelLoader::load_from_bytes(&model_bytes);
    // May fail due to invalid trailing data, but should not panic
    let _ = result;
}
