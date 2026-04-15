//! Backend abstraction for compute dispatch.
//!
//! The [`KernelBackend`] trait defines the operations the transformer
//! forward pass needs. Implementations exist for CPU (via `nnx-kernels`)
//! and GPU (via `nnx-cubecl`/CubeCL).

/// Abstraction over CPU and GPU compute backends.
///
/// The transformer execution is generic over this trait, allowing
/// the same model forward pass to run on CPU (via nnx-kernels) or
/// GPU (via nnx-cubecl/CubeCL).
pub trait KernelBackend: Send + Sync {
    /// Opaque buffer type for this backend (e.g., `Vec<f32>` for CPU,
    /// GPU handle for CubeCL).
    type Buffer: Send + Sync;

    // --- Allocation ---

    /// Create a buffer from f32 data.
    fn from_f32(&self, data: &[f32]) -> Self::Buffer;

    /// Read buffer contents back to f32.
    fn to_f32(&self, buffer: &Self::Buffer) -> Vec<f32>;

    /// Create a zero-filled buffer of `len` elements.
    fn zeros(&self, len: usize) -> Self::Buffer;

    // --- Matrix operations ---

    /// Matrix-vector multiply: y = A @ x, where A is [m, k] row-major.
    fn matvec(
        &self,
        matrix: &Self::Buffer,
        x: &Self::Buffer,
        y: &mut Self::Buffer,
        m: usize,
        k: usize,
    );

    /// Matrix-vector multiply with bias: y = A @ x + bias, where A is [m, k] row-major.
    fn matvec_bias(
        &self,
        matrix: &Self::Buffer,
        x: &Self::Buffer,
        bias: &Self::Buffer,
        y: &mut Self::Buffer,
        m: usize,
        k: usize,
    );

    /// Dot product of two buffers.
    fn dot(&self, a: &Self::Buffer, b: &Self::Buffer) -> f32;

    // --- Normalization ---

    /// RMS normalization: output = (x / rms(x)) * weight.
    fn rms_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        eps: f32,
    );

    /// Layer normalization (without bias).
    fn layer_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        hidden_dim: usize,
        eps: f32,
    );

    /// Layer normalization (with bias).
    fn layer_norm_bias(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        bias: &Self::Buffer,
        output: &mut Self::Buffer,
        hidden_dim: usize,
        eps: f32,
    );

    // --- Activations ---

    /// SiLU activation in-place.
    fn silu_inplace(&self, x: &mut Self::Buffer);

    /// GELU activation in-place.
    fn gelu_inplace(&self, x: &mut Self::Buffer);

    /// Elementwise multiply in-place: a *= b.
    fn mul_inplace(&self, a: &mut Self::Buffer, b: &Self::Buffer);

    /// Elementwise add in-place: a += b.
    fn add_inplace(&self, a: &mut Self::Buffer, b: &Self::Buffer);

    /// Fused SwiGLU: gate = silu(gate) * up.
    fn fused_swiglu(&self, gate: &mut Self::Buffer, up: &Self::Buffer);

    /// Fused GeGLU: gate = gelu(gate) * up.
    fn fused_geglu(&self, gate: &mut Self::Buffer, up: &Self::Buffer);

    // --- Position encoding ---

    /// Apply RoPE in-place to a single head's data.
    fn rope_inplace(
        &self,
        data: &mut Self::Buffer,
        head_offset: usize,
        head_dim: usize,
        position: usize,
        freq_base: f32,
    );

    /// Apply RoPE in-place to only the first `rotary_dim` dimensions of a head.
    fn partial_rope_inplace(
        &self,
        data: &mut Self::Buffer,
        head_offset: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        freq_base: f32,
    );

    // --- Attention ---

    /// Softmax in-place over a row identified by `offset` and `len`.
    fn softmax_inplace(&self, data: &mut Self::Buffer, offset: usize, len: usize);
}
