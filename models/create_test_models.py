#!/usr/bin/env python3
"""
Create simple synthetic ONNX models for integration testing.

These are minimal valid ONNX models that can be used to test the
RONN integration pipeline without requiring downloads or large models.
"""

import sys

def create_simple_linear_model():
    """Create a simple linear transformation model: y = Wx + b"""
    try:
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
    except ImportError:
        print("Error: onnx package not found")
        print("Install with: pip install onnx numpy")
        return False

    # Define input
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])

    # Define output
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

    # Create weights (4x4 matrix)
    W = helper.make_tensor(
        name='W',
        data_type=TensorProto.FLOAT,
        dims=[4, 4],
        vals=np.eye(4, dtype=np.float32).flatten().tolist()  # Identity matrix
    )

    # Create bias (4 elements)
    b = helper.make_tensor(
        name='b',
        data_type=TensorProto.FLOAT,
        dims=[4],
        vals=[0.0, 0.0, 0.0, 0.0]
    )

    # Create MatMul node
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input', 'W'],
        outputs=['matmul_out']
    )

    # Create Add node (for bias)
    add_node = helper.make_node(
        'Add',
        inputs=['matmul_out', 'b'],
        outputs=['output']
    )

    # Create graph
    graph_def = helper.make_graph(
        [matmul_node, add_node],
        'simple_linear',
        [X],
        [Y],
        [W, b]
    )

    # Create model
    model_def = helper.make_model(graph_def, producer_name='ronn-test')
    model_def.opset_import[0].version = 13

    # Check and save
    onnx.checker.check_model(model_def)
    onnx.save(model_def, 'simple_linear.onnx')

    print("✓ Created simple_linear.onnx (linear transformation)")
    return True


def create_simple_conv_model():
    """Create a simple 2D convolution model for image processing"""
    try:
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
    except ImportError:
        print("Error: onnx package not found")
        return False

    # Input: [batch=1, channels=3, height=224, width=224]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])

    # Output: [batch=1, channels=64, height=112, width=112]
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 64, 112, 112])

    # Conv kernel: [out_channels=64, in_channels=3, kH=7, kW=7]
    kernel_shape = [64, 3, 7, 7]
    kernel_size = np.prod(kernel_shape)
    W = helper.make_tensor(
        name='conv_weight',
        data_type=TensorProto.FLOAT,
        dims=kernel_shape,
        vals=np.random.randn(kernel_size).astype(np.float32).tolist()
    )

    # Bias: [64]
    b = helper.make_tensor(
        name='conv_bias',
        data_type=TensorProto.FLOAT,
        dims=[64],
        vals=np.zeros(64, dtype=np.float32).tolist()
    )

    # Create Conv node
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'conv_weight', 'conv_bias'],
        outputs=['conv_out'],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
    )

    # Create ReLU node
    relu_node = helper.make_node(
        'Relu',
        inputs=['conv_out'],
        outputs=['output']
    )

    # Create graph
    graph_def = helper.make_graph(
        [conv_node, relu_node],
        'simple_conv',
        [X],
        [Y],
        [W, b]
    )

    # Create model
    model_def = helper.make_model(graph_def, producer_name='ronn-test')
    model_def.opset_import[0].version = 13

    # Check and save
    onnx.checker.check_model(model_def)
    onnx.save(model_def, 'simple_conv.onnx')

    print("✓ Created simple_conv.onnx (conv + relu)")
    return True


def create_resnet_stub():
    """Create a stub ResNet-18 model (simplified version)"""
    try:
        import onnx
        from onnx import helper, TensorProto
        import numpy as np
    except ImportError:
        return False

    # Input: [batch=1, channels=3, height=224, width=224]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])

    # Output: [batch=1, classes=1000]
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])

    # Very simplified ResNet-like structure
    # Just conv -> pool -> flatten -> linear to demonstrate the pipeline

    # Conv layer
    conv_w = helper.make_tensor(
        'conv_w',
        TensorProto.FLOAT,
        [64, 3, 3, 3],
        np.random.randn(64 * 3 * 3 * 3).astype(np.float32).tolist()
    )

    conv_node = helper.make_node(
        'Conv',
        ['input', 'conv_w'],
        ['conv_out'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1]
    )

    # Global average pooling
    pool_node = helper.make_node(
        'GlobalAveragePool',
        ['conv_out'],
        ['pool_out']
    )

    # Reshape to [1, 64]
    shape_const = helper.make_tensor(
        'shape',
        TensorProto.INT64,
        [2],
        [1, 64]
    )

    reshape_node = helper.make_node(
        'Reshape',
        ['pool_out', 'shape'],
        ['flat']
    )

    # Final linear layer to 1000 classes
    fc_w = helper.make_tensor(
        'fc_w',
        TensorProto.FLOAT,
        [64, 1000],
        np.random.randn(64 * 1000).astype(np.float32).tolist()
    )

    fc_node = helper.make_node(
        'MatMul',
        ['flat', 'fc_w'],
        ['output']
    )

    # Create graph
    graph_def = helper.make_graph(
        [conv_node, pool_node, reshape_node, fc_node],
        'resnet18_stub',
        [X],
        [Y],
        [conv_w, shape_const, fc_w]
    )

    # Create model
    model_def = helper.make_model(graph_def, producer_name='ronn-test')
    model_def.opset_import[0].version = 13

    # Check and save
    onnx.checker.check_model(model_def)
    onnx.save(model_def, 'resnet18.onnx')

    print("✓ Created resnet18.onnx (simplified stub)")
    return True


def main():
    print("Creating synthetic ONNX models for testing...")
    print()

    success_count = 0

    if create_simple_linear_model():
        success_count += 1

    if create_simple_conv_model():
        success_count += 1

    if create_resnet_stub():
        success_count += 1

    print()
    print(f"Created {success_count}/3 models successfully")

    if success_count < 3:
        print()
        print("Note: Install required packages:")
        print("  pip install onnx numpy")

    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
