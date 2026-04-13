//! Optimized compute kernels for NNX.
//!
//! Pure Rust implementations of all operations needed for neural network
//! inference. SIMD-accelerated where possible, rayon-parallelized for
//! large workloads.
//!
//! ## Operator Coverage
//!
//! | Category | Operators |
//! |----------|-----------|
//! | Activation | ReLU, Sigmoid, Tanh, GELU, SiLU/Swish, LeakyReLU, ELU, Softmax, LogSoftmax |
//! | Element-wise | Add, Sub, Mul, Div, Neg, Abs, Sqrt, Exp, Log, Pow, Clip, Floor, Ceil |
//! | Matrix | MatMul, MatVec, Dot, Gemm |
//! | Normalization | RMSNorm, LayerNorm, BatchNorm |
//! | Pooling | MaxPool2D, AvgPool2D, GlobalAvgPool |
//! | Convolution | Conv2D |
//! | Reduction | Sum, Mean, Max, Min, ArgMax, ArgMin, Prod |
//! | Shape | Reshape, Transpose, Concat, Split, Gather, Slice, Squeeze, Unsqueeze, Pad, Flatten |
//! | Position | RoPE |
//! | Sampling | Softmax (also in activation), TopK |

pub mod activations;
pub mod conv;
pub mod elementwise;
pub mod matmul;
pub mod normalization;
pub mod pooling;
pub mod reduction;
pub mod rope;
pub mod rms_norm;
pub mod shape_ops;
pub mod softmax;
