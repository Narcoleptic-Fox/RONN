//! Unit tests for ONNX model loader
//! Tests: Model parsing, graph conversion, attribute extraction, initializer loading

use prost::Message;
use ronn_core::NodeAttribute;
use ronn_onnx::ModelLoader;
use ronn_onnx::onnx_proto::*;

// Helper functions to create common protobuf structures with less boilerplate
fn dim_value(val: i64) -> tensor_shape_proto::Dimension {
    tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(tensor_shape_proto::dimension::Value::DimValue(val)),
    }
}

fn dim_param(name: &str) -> tensor_shape_proto::Dimension {
    tensor_shape_proto::Dimension {
        denotation: String::new(),
        value: Some(tensor_shape_proto::dimension::Value::DimParam(
            name.to_string(),
        )),
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

// ============ Model Loading Tests ============

#[test]
fn test_load_minimal_model() {
    // Create minimal valid ONNX model
    let model_proto = ModelProto {
        ir_version: 7,
        producer_name: "test".to_string(),
        graph: Some(GraphProto {
            name: "test_graph".to_string(),
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

    assert!(result.is_ok(), "Should load minimal model");

    let model = result.unwrap();
    assert_eq!(model.ir_version, 7);
    assert_eq!(model.producer_name, Some("test".to_string()));
}

#[test]
fn test_load_model_with_nodes() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test_graph".to_string(),
            node: vec![NodeProto {
                name: "relu_node".to_string(),
                op_type: "Relu".to_string(),
                input: vec!["input1".to_string()],
                output: vec!["output1".to_string()],
                attribute: vec![],
                ..Default::default()
            }],
            input: vec![value_info("input1", 1, vec![1, 3, 224, 224])],
            output: vec![value_info("output1", 1, vec![1, 3, 224, 224])],
            initializer: vec![],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.inputs[0].name, "input1");
    assert_eq!(model.inputs[0].shape, vec![1, 3, 224, 224]);

    assert_eq!(model.outputs.len(), 1);
    assert_eq!(model.outputs[0].name, "output1");
}

#[test]
fn test_load_model_with_initializers() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test_graph".to_string(),
            node: vec![],
            input: vec![
                value_info("input1", 1, vec![2]),
                value_info("weight", 1, vec![2]),
            ],
            output: vec![],
            initializer: vec![TensorProto {
                name: "weight".to_string(),
                dims: vec![2],
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

    assert!(result.is_ok());

    let model = result.unwrap();
    // Weight should be in initializers, not in inputs
    assert_eq!(model.inputs.len(), 1);
    assert_eq!(model.inputs[0].name, "input1");

    assert!(model.initializers.contains_key("weight"));
    assert_eq!(model.initializers.len(), 1);
}

#[test]
fn test_load_initializer_from_raw_data_f32() {
    let raw_data = [1.0_f32, 2.5_f32]
        .into_iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>();

    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test_graph".to_string(),
            node: vec![],
            input: vec![value_info("input1", 1, vec![2])],
            output: vec![],
            initializer: vec![TensorProto {
                name: "weight".to_string(),
                dims: vec![2],
                data_type: 1, // FLOAT
                raw_data,
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let model = ModelLoader::load_from_bytes(&model_bytes).unwrap();
    let weight = model.initializers.get("weight").unwrap();
    let values = weight.to_vec().unwrap();

    assert_eq!(values, vec![1.0, 2.5]);
}

#[test]
fn test_load_invalid_json() {
    let invalid_json = b"{ invalid json }";
    let result = ModelLoader::load_from_bytes(invalid_json);

    assert!(result.is_err());
}

#[test]
fn test_load_model_without_graph() {
    let model_proto = ModelProto {
        ir_version: 7,
        producer_name: "test".to_string(),
        graph: None, // Missing graph field
        ..Default::default()
    };

    let model_bytes = model_proto.encode_to_vec();
    let result = ModelLoader::load_from_bytes(&model_bytes);

    assert!(result.is_err(), "Should fail without graph");
}

// ============ Attribute Parsing Tests ============

#[test]
fn test_parse_float_attribute() {
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
                    name: "alpha".to_string(),
                    f: 0.5,
                    r#type: attribute_proto::AttributeType::Float as i32,
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

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    assert_eq!(nodes.len(), 1);

    let node = &nodes[0];
    assert!(node.attributes.contains_key("alpha"));
    if let Some(NodeAttribute::Float(val)) = node.attributes.get("alpha") {
        assert!((val - 0.5).abs() < 1e-6);
    } else {
        panic!("Expected Float attribute");
    }
}

#[test]
fn test_parse_int_attribute() {
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
                    name: "axis".to_string(),
                    i: 1,
                    r#type: attribute_proto::AttributeType::Int as i32,
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

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    if let Some(NodeAttribute::Int(val)) = node.attributes.get("axis") {
        assert_eq!(*val, 1);
    } else {
        panic!("Expected Int attribute");
    }
}

#[test]
fn test_parse_string_attribute() {
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
                    name: "mode".to_string(),
                    s: b"test".to_vec(),
                    r#type: attribute_proto::AttributeType::String as i32,
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

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    if let Some(NodeAttribute::String(val)) = node.attributes.get("mode") {
        assert_eq!(val, "test");
    } else {
        panic!("Expected String attribute");
    }
}

#[test]
fn test_parse_floats_attribute() {
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
                    name: "scales".to_string(),
                    floats: vec![1.0, 2.0, 3.0],
                    r#type: attribute_proto::AttributeType::Floats as i32,
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

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    if let Some(NodeAttribute::FloatArray(vals)) = node.attributes.get("scales") {
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 2.0).abs() < 1e-6);
        assert!((vals[2] - 3.0).abs() < 1e-6);
    } else {
        panic!("Expected FloatArray attribute");
    }
}

#[test]
fn test_parse_ints_attribute() {
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
                    name: "pads".to_string(),
                    ints: vec![1, 1, 1, 1],
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

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    if let Some(NodeAttribute::IntArray(vals)) = node.attributes.get("pads") {
        assert_eq!(vals, &[1, 1, 1, 1]);
    } else {
        panic!("Expected IntArray attribute");
    }
}

#[test]
fn test_parse_multiple_attributes() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "conv_node".to_string(),
                op_type: "Conv".to_string(),
                input: vec!["input".to_string(), "weight".to_string()],
                output: vec!["output".to_string()],
                attribute: vec![
                    AttributeProto {
                        name: "group".to_string(),
                        i: 1,
                        r#type: attribute_proto::AttributeType::Int as i32,
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "strides".to_string(),
                        ints: vec![1, 1],
                        r#type: attribute_proto::AttributeType::Ints as i32,
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "pads".to_string(),
                        ints: vec![0, 0, 0, 0],
                        r#type: attribute_proto::AttributeType::Ints as i32,
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

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();
    let node = &nodes[0];

    assert_eq!(node.attributes.len(), 3);
    assert!(node.attributes.contains_key("group"));
    assert!(node.attributes.contains_key("strides"));
    assert!(node.attributes.contains_key("pads"));
}

// ============ Shape Inference Tests ============

#[test]
fn test_shape_inference_static() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![],
            input: vec![value_info("input", 1, vec![1, 3, 224, 224])],
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
    assert_eq!(model.inputs[0].shape, vec![1, 3, 224, 224]);
}

#[test]
fn test_shape_inference_dynamic() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![],
            input: vec![ValueInfoProto {
                name: "input".to_string(),
                r#type: Some(TypeProto {
                    denotation: String::new(),
                    value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                        elem_type: 1,
                        shape: Some(TensorShapeProto {
                            dim: vec![
                                dim_param("batch"),
                                dim_value(3),
                                dim_value(224),
                                dim_value(224),
                            ],
                        }),
                    })),
                }),
                ..Default::default()
            }],
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
    // Dynamic dimensions are represented as 0
    assert_eq!(model.inputs[0].shape, vec![0, 3, 224, 224]);
}

// ============ Graph Structure Tests ============

#[test]
fn test_node_name_generation() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![NodeProto {
                name: "".to_string(), // Node without name - should be auto-generated
                op_type: "Relu".to_string(),
                input: vec!["x".to_string()],
                output: vec!["y".to_string()],
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

    assert!(result.is_ok());

    let model = result.unwrap();
    let nodes = model.graph.nodes();

    // Should have generated a name like "node_0"
    assert!(nodes[0].name.is_some());
}

#[test]
fn test_multiple_nodes() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "test".to_string(),
            node: vec![
                NodeProto {
                    name: "relu1".to_string(),
                    op_type: "Relu".to_string(),
                    input: vec!["input".to_string()],
                    output: vec!["relu_out".to_string()],
                    attribute: vec![],
                    ..Default::default()
                },
                NodeProto {
                    name: "add1".to_string(),
                    op_type: "Add".to_string(),
                    input: vec!["relu_out".to_string(), "bias".to_string()],
                    output: vec!["output".to_string()],
                    attribute: vec![],
                    ..Default::default()
                },
            ],
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

    let model = result.unwrap();
    let nodes = model.graph.nodes();

    assert_eq!(nodes.len(), 2);
    assert_eq!(nodes[0].op_type, "Relu");
    assert_eq!(nodes[1].op_type, "Add");
}

// ============ Version Compatibility Tests ============

#[test]
fn test_various_ir_versions() {
    let versions = vec![3, 4, 5, 6, 7, 8, 9];

    for version in versions {
        let model_proto = ModelProto {
            ir_version: version,
            graph: Some(GraphProto {
                name: "test".to_string(),
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

        assert!(result.is_ok(), "Should support IR version {}", version);

        let model = result.unwrap();
        assert_eq!(model.ir_version, version);
    }
}

// ============ Edge Cases ============

#[test]
fn test_empty_graph() {
    let model_proto = ModelProto {
        ir_version: 7,
        graph: Some(GraphProto {
            name: "empty".to_string(),
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

    assert!(result.is_ok());

    let model = result.unwrap();
    assert_eq!(model.inputs.len(), 0);
    assert_eq!(model.outputs.len(), 0);
    let nodes = model.graph.nodes();
    assert_eq!(nodes.len(), 0);
}

#[test]
fn test_producer_name_optional() {
    let model_proto = ModelProto {
        ir_version: 7,
        producer_name: "".to_string(), // Empty producer name
        graph: Some(GraphProto {
            name: "test".to_string(),
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

    assert!(result.is_ok());

    let model = result.unwrap();
    // Empty string might be treated as None
    assert!(model.producer_name.is_none() || model.producer_name == Some("".to_string()));
}
