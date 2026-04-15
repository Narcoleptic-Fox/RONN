//! CubeCL GPU implementation of [`KernelBackend`].
//!
//! [`CubeclBackend`] wraps a CubeCL `ComputeClient` and dispatches every
//! trait method to the GPU kernel in the corresponding sibling module.
//! The backend is generic over `R: Runtime`, so callers choose the
//! concrete GPU API (CUDA, ROCm, Metal, Vulkan, WebGPU) at instantiation.

use cubecl::prelude::*;
use cubecl::server::Handle;
use nnx_core::backend::KernelBackend;

use crate::activations::BLOCK_SIZE;

/// Compute tiled launch configuration for elementwise kernels.
/// Returns (CubeCount, CubeDim) where each cube has BLOCK_SIZE threads
/// and enough cubes to cover `len` elements.
fn tiled_launch(len: usize) -> (CubeCount, CubeDim) {
    let block = BLOCK_SIZE;
    let num_cubes = (len as u32 + block - 1) / block;
    (
        CubeCount::Static(num_cubes, 1, 1),
        CubeDim::new(block, 1, 1),
    )
}

/// Opaque GPU buffer: a server-side handle plus element count.
///
/// The handle is reference-counted internally by CubeCL, so cloning a
/// `GpuBuffer` is cheap (it does not duplicate device memory).
#[derive(Clone)]
pub struct GpuBuffer {
    pub handle: Handle,
    pub len: usize,
}

// Safety: CubeCL handles are internally synchronized by the compute server.
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

/// GPU compute backend powered by CubeCL.
///
/// Generic over `R: Runtime` — the caller picks the concrete GPU API.
///
/// # Example
///
/// ```ignore
/// use cubecl::wgpu::WgpuRuntime;
/// let backend = CubeclBackend::<WgpuRuntime>::new();
/// ```
pub struct CubeclBackend<R: Runtime> {
    client: ComputeClient<<R as Runtime>::Server>,
}

// Manual Send + Sync: ComputeClient is internally thread-safe.
unsafe impl<R: Runtime> Send for CubeclBackend<R> {}
unsafe impl<R: Runtime> Sync for CubeclBackend<R> {}

impl<R: Runtime> CubeclBackend<R> {
    fn encode_u32(data: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * std::mem::size_of::<u32>());
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decode_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    /// Create a backend using the default device for this runtime.
    pub fn new() -> Self {
        let device = <R as Runtime>::Device::default();
        let client = R::client(&device);
        Self { client }
    }

    /// Create a backend targeting a specific device.
    pub fn with_device(device: &<R as Runtime>::Device) -> Self {
        let client = R::client(device);
        Self { client }
    }

    /// Get a reference to the underlying compute client.
    ///
    /// Used by [`crate::inference::GpuInference`] to launch custom attention
    /// kernels directly.
    pub fn client(&self) -> &ComputeClient<<R as Runtime>::Server> {
        &self.client
    }

    /// Upload f32 data to the GPU and return a handle.
    fn upload(&self, data: &[f32]) -> Handle {
        self.client.create(f32::as_bytes(data))
    }

    /// Download f32 data from the GPU.
    fn download(&self, handle: &Handle) -> Vec<f32> {
        let bytes = self.client.read_one(handle.clone());
        f32::from_bytes(&bytes).to_vec()
    }

    /// Upload u32 data to the GPU and return a handle.
    pub fn from_u32(&self, data: &[u32]) -> GpuBuffer {
        let bytes = Self::encode_u32(data);
        GpuBuffer {
            handle: self.client.create(&bytes),
            len: data.len(),
        }
    }

    /// Download u32 data from the GPU.
    pub fn to_u32(&self, buffer: &GpuBuffer) -> Vec<u32> {
        let bytes = self.client.read_one(buffer.handle.clone());
        Self::decode_u32(&bytes)
    }
}

impl<R: Runtime> KernelBackend for CubeclBackend<R> {
    type Buffer = GpuBuffer;

    // --- Allocation ---

    fn from_f32(&self, data: &[f32]) -> Self::Buffer {
        GpuBuffer {
            handle: self.upload(data),
            len: data.len(),
        }
    }

    fn to_f32(&self, buffer: &Self::Buffer) -> Vec<f32> {
        self.download(&buffer.handle)
    }

    fn zeros(&self, len: usize) -> Self::Buffer {
        let data = vec![0.0f32; len];
        GpuBuffer {
            handle: self.upload(&data),
            len,
        }
    }

    // --- Matrix operations ---

    fn matvec(
        &self,
        matrix: &Self::Buffer,
        x: &Self::Buffer,
        y: &mut Self::Buffer,
        m: usize,
        k: usize,
    ) {
        unsafe {
            crate::matmul::matvec_kernel::launch::<R>(
                &self.client,
                CubeCount::Static(m as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&matrix.handle, m * k, 1),
                ArrayArg::from_raw_parts::<f32>(&x.handle, k, 1),
                ArrayArg::from_raw_parts::<f32>(&y.handle, m, 1),
                ScalarArg::new(k as u32),
            );
        }
    }

    fn dot(&self, a: &Self::Buffer, b: &Self::Buffer) -> f32 {
        let len = a.len;
        let out = self.zeros(1);

        unsafe {
            crate::matmul::dot_kernel::launch::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&a.handle, len, 1),
                ArrayArg::from_raw_parts::<f32>(&b.handle, len, 1),
                ArrayArg::from_raw_parts::<f32>(&out.handle, 1, 1),
                ScalarArg::new(len as u32),
            );
        }

        let result = self.download(&out.handle);
        result[0]
    }

    // --- Normalization ---

    fn rms_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        eps: f32,
    ) {
        let hidden_dim = x.len;

        unsafe {
            crate::normalization::rms_norm_kernel::launch::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&x.handle, hidden_dim, 1),
                ArrayArg::from_raw_parts::<f32>(&weight.handle, hidden_dim, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, hidden_dim, 1),
                ScalarArg::new(hidden_dim as u32),
                ScalarArg::new(eps),
            );
        }
    }

    fn layer_norm(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        output: &mut Self::Buffer,
        hidden_dim: usize,
        eps: f32,
    ) {
        let num_vectors = x.len / hidden_dim;

        unsafe {
            crate::normalization::layer_norm_kernel::launch::<R>(
                &self.client,
                CubeCount::Static(num_vectors as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&x.handle, x.len, 1),
                ArrayArg::from_raw_parts::<f32>(&weight.handle, hidden_dim, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, x.len, 1),
                ScalarArg::new(hidden_dim as u32),
                ScalarArg::new(eps),
            );
        }
    }

    fn layer_norm_bias(
        &self,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        bias: &Self::Buffer,
        output: &mut Self::Buffer,
        hidden_dim: usize,
        eps: f32,
    ) {
        let num_vectors = x.len / hidden_dim;

        unsafe {
            crate::normalization::layer_norm_bias_kernel::launch::<R>(
                &self.client,
                CubeCount::Static(num_vectors as u32, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&x.handle, x.len, 1),
                ArrayArg::from_raw_parts::<f32>(&weight.handle, hidden_dim, 1),
                ArrayArg::from_raw_parts::<f32>(&bias.handle, hidden_dim, 1),
                ArrayArg::from_raw_parts::<f32>(&output.handle, x.len, 1),
                ScalarArg::new(hidden_dim as u32),
                ScalarArg::new(eps),
            );
        }
    }

    // --- Activations ---
    // All elementwise ops use tiled launches: BLOCK_SIZE threads per cube,
    // ceil(len / BLOCK_SIZE) cubes. Kernels bounds-check via `if idx < len`.

    fn silu_inplace(&self, x: &mut Self::Buffer) {
        let len = x.len;
        let (cubes, block) = tiled_launch(len);

        unsafe {
            crate::activations::silu_inplace_kernel::launch::<R>(
                &self.client,
                cubes,
                block,
                ArrayArg::from_raw_parts::<f32>(&x.handle, len, 1),
                ScalarArg::new(len as u32),
            );
        }
    }

    fn gelu_inplace(&self, x: &mut Self::Buffer) {
        let len = x.len;
        let (cubes, block) = tiled_launch(len);

        unsafe {
            crate::activations::gelu_inplace_kernel::launch::<R>(
                &self.client,
                cubes,
                block,
                ArrayArg::from_raw_parts::<f32>(&x.handle, len, 1),
                ScalarArg::new(len as u32),
            );
        }
    }

    fn mul_inplace(&self, a: &mut Self::Buffer, b: &Self::Buffer) {
        let len = a.len;
        let (cubes, block) = tiled_launch(len);

        unsafe {
            crate::activations::mul_inplace_kernel::launch::<R>(
                &self.client,
                cubes,
                block,
                ArrayArg::from_raw_parts::<f32>(&a.handle, len, 1),
                ArrayArg::from_raw_parts::<f32>(&b.handle, len, 1),
                ScalarArg::new(len as u32),
            );
        }
    }

    fn add_inplace(&self, a: &mut Self::Buffer, b: &Self::Buffer) {
        let len = a.len;
        let (cubes, block) = tiled_launch(len);

        unsafe {
            crate::activations::add_inplace_kernel::launch::<R>(
                &self.client,
                cubes,
                block,
                ArrayArg::from_raw_parts::<f32>(&a.handle, len, 1),
                ArrayArg::from_raw_parts::<f32>(&b.handle, len, 1),
                ScalarArg::new(len as u32),
            );
        }
    }

    // --- Position encoding ---

    fn rope_inplace(
        &self,
        data: &mut Self::Buffer,
        head_offset: usize,
        head_dim: usize,
        position: usize,
        freq_base: f32,
    ) {
        unsafe {
            crate::rope::rope_kernel::launch::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&data.handle, data.len, 1),
                ScalarArg::new(head_offset as u32),
                ScalarArg::new(head_dim as u32),
                ScalarArg::new(position as u32),
                ScalarArg::new(freq_base),
            );
        }
    }

    // --- Attention ---

    fn softmax_inplace(&self, data: &mut Self::Buffer, offset: usize, len: usize) {
        unsafe {
            crate::softmax::softmax_kernel::launch::<R>(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new(1, 1, 1),
                ArrayArg::from_raw_parts::<f32>(&data.handle, data.len, 1),
                ScalarArg::new(offset as u32),
                ScalarArg::new(len as u32),
            );
        }
    }
}
