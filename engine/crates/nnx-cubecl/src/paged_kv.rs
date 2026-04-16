//! GPU-resident paged KV cache storage and paged attention kernels.
//!
//! The backing storage is a single flat GPU buffer where page `i` starts at:
//! `i * page_size * 2 * num_kv_heads * head_dim`.
//! Each page preserves the CPU paged-cache layout:
//! `[token][K|V][kv_head][head_dim]`.

use cubecl::prelude::*;
use nnx_core::backend::KernelBackend;
use nnx_core::PageId;

use crate::backend::{CubeclBackend, GpuBuffer};

const KV_QJL_SKETCH_DIM: u32 = 16;
const KV_QJL_SKETCH_DIM_F32: f32 = 16.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuPagedKvQuantConfig {
    pub residual_sketch_dim: usize,
}

impl GpuPagedKvQuantConfig {
    pub const SUPPORTED_RESIDUAL_SKETCH_DIM: usize = KV_QJL_SKETCH_DIM as usize;

    pub fn validate(self) {
        assert_eq!(
            self.residual_sketch_dim,
            Self::SUPPORTED_RESIDUAL_SKETCH_DIM,
            "GpuPagedKvQuantConfig.residual_sketch_dim={} is unsupported; the current CubeCL quantized paged-KV path requires {} sketch dimensions",
            self.residual_sketch_dim,
            Self::SUPPORTED_RESIDUAL_SKETCH_DIM
        );
    }

    pub fn residual_words_per_token(self) -> usize {
        self.validate();
        self.residual_sketch_dim.div_ceil(32)
    }
}

pub struct GpuPagedKvQuantizedStorage {
    config: GpuPagedKvQuantConfig,
    page_flags: Vec<u32>,
    quant_payload: GpuBuffer,
}

impl GpuPagedKvQuantizedStorage {
    pub fn config(&self) -> GpuPagedKvQuantConfig {
        self.config
    }

    pub fn is_quantized(&self, id: PageId) -> bool {
        self.page_flags[id.0 as usize] != 0
    }

    pub fn quant_payload(&self) -> &GpuBuffer {
        &self.quant_payload
    }

    fn set_quantized(&mut self, id: PageId) {
        self.page_flags[id.0 as usize] = 1;
    }

    fn set_dense(&mut self, id: PageId) {
        self.page_flags[id.0 as usize] = 0;
    }
}

/// One logical GPU page within a flat page-pool allocation.
///
/// `data` references the page-pool allocation, so callers must pair it with
/// `base_offset()` when launching kernels that operate on a specific page.
#[derive(Clone)]
pub struct GpuPhysicalPage {
    data: GpuBuffer,
    base_offset: usize,
    page_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl GpuPhysicalPage {
    /// Allocate a standalone GPU page.
    pub fn new<R: Runtime>(
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        backend: &CubeclBackend<R>,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        let page_len = page_size * 2 * kv_dim;
        Self {
            data: backend.zeros(page_len),
            base_offset: 0,
            page_size,
            num_kv_heads,
            head_dim,
        }
    }

    fn from_buffer_view(
        data: GpuBuffer,
        base_offset: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            data,
            base_offset,
            page_size,
            num_kv_heads,
            head_dim,
        }
    }

    /// Underlying pool allocation handle.
    pub fn data(&self) -> &GpuBuffer {
        &self.data
    }

    /// Base offset into the flat page-pool buffer, in f32 elements.
    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    /// Tokens per page.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Number of KV heads stored in this page.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// KV dimension per token: `num_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Slot size per token: `2 * kv_dim`.
    pub fn slot_size(&self) -> usize {
        2 * self.kv_dim()
    }

    /// Total page length in f32 elements.
    pub fn len(&self) -> usize {
        self.page_size * self.slot_size()
    }

    /// Memory usage in bytes for this page.
    pub fn memory_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<f32>()
    }
}

/// Fixed GPU pool of paged KV storage.
///
/// Page ownership and ref-counting stay on the CPU in `BlockAllocator`.
/// This struct only provides GPU access to the flat backing allocation and
/// page-local metadata/views.
pub struct GpuPagePool {
    flat_buffer: GpuBuffer,
    pages: Vec<GpuPhysicalPage>,
    page_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    quantized: Option<GpuPagedKvQuantizedStorage>,
}

impl GpuPagePool {
    /// Create a flat GPU page pool with `num_pages` pages.
    pub fn new<R: Runtime>(
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        backend: &CubeclBackend<R>,
    ) -> Self {
        Self::new_with_quantization(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            None,
            backend,
        )
    }

    pub fn new_with_quantization<R: Runtime>(
        num_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        quant_config: Option<GpuPagedKvQuantConfig>,
        backend: &CubeclBackend<R>,
    ) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        let page_stride = page_size * 2 * kv_dim;
        if let Some(config) = quant_config {
            config.validate();
            let metadata_words_per_page = 4 * kv_dim + page_size * num_kv_heads;
            assert!(
                page_stride >= metadata_words_per_page,
                "quantized paged KV requires page_size large enough to hold per-page metadata: page_stride={} metadata_words_per_page={} page_size={} num_kv_heads={} head_dim={}",
                page_stride,
                metadata_words_per_page,
                page_size,
                num_kv_heads,
                head_dim
            );
        }
        let flat_buffer = backend.zeros(num_pages * page_stride);
        let pages = (0..num_pages)
            .map(|page_idx| {
                GpuPhysicalPage::from_buffer_view(
                    flat_buffer.clone(),
                    page_idx * page_stride,
                    page_size,
                    num_kv_heads,
                    head_dim,
                )
            })
            .collect();

        let quantized = quant_config.map(|config| {
            let words_per_token = kv_dim.div_ceil(4);
            let residual_words_per_token = config.residual_words_per_token();
            let residual_words_per_page = page_size * num_kv_heads * residual_words_per_token;
            let payload_words_per_page =
                2 * page_size * words_per_token + residual_words_per_page;

            GpuPagedKvQuantizedStorage {
                config,
                page_flags: vec![0u32; num_pages],
                quant_payload: backend.from_u32(&vec![0u32; num_pages * payload_words_per_page]),
            }
        });

        Self {
            flat_buffer,
            pages,
            page_size,
            num_kv_heads,
            head_dim,
            quantized,
        }
    }

    /// Get a page view by physical page id.
    pub fn page(&self, id: PageId) -> &GpuPhysicalPage {
        &self.pages[id.0 as usize]
    }

    /// Get a mutable page view by physical page id.
    pub fn page_mut(&mut self, id: PageId) -> &mut GpuPhysicalPage {
        &mut self.pages[id.0 as usize]
    }

    /// Flat backing allocation shared by all pages.
    pub fn flat_buffer(&self) -> &GpuBuffer {
        &self.flat_buffer
    }

    pub fn replace_flat_buffer(&mut self, flat_buffer: GpuBuffer) {
        let page_stride = self.page_stride();
        let num_pages = self.pages.len();
        self.flat_buffer = flat_buffer.clone();
        self.pages = (0..num_pages)
            .map(|page_idx| {
                GpuPhysicalPage::from_buffer_view(
                    flat_buffer.clone(),
                    page_idx * page_stride,
                    self.page_size,
                    self.num_kv_heads,
                    self.head_dim,
                )
            })
            .collect();
    }

    /// Number of pages in the pool.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Tokens per page.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Number of KV heads per page.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// KV dimension: `num_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Per-page stride within the flat buffer, in f32 elements.
    pub fn page_stride(&self) -> usize {
        self.page_size * 2 * self.kv_dim()
    }

    pub fn quantized(&self) -> Option<&GpuPagedKvQuantizedStorage> {
        self.quantized.as_ref()
    }

    pub fn is_page_quantized(&self, id: PageId) -> bool {
        self.quantized
            .as_ref()
            .map(|storage| storage.is_quantized(id))
            .unwrap_or(false)
    }

    pub fn set_page_dense(&mut self, id: PageId) {
        if let Some(quantized) = self.quantized.as_mut() {
            quantized.set_dense(id);
        }
    }

    pub fn set_page_quantized(&mut self, id: PageId) {
        if let Some(quantized) = self.quantized.as_mut() {
            quantized.set_quantized(id);
        }
    }

    pub fn replace_quant_payload(&mut self, quant_payload: GpuBuffer) {
        if let Some(quantized) = self.quantized.as_mut() {
            quantized.quant_payload = quant_payload;
        }
    }
}

/// Quantize one completed page in place.
///
/// The page's dense K/V payload is read from `page_data`, quantized payload and
/// residual sketch bits are written into `quant_payload`, and the page's dense
/// storage is then repurposed to hold quantization metadata.
#[cube(launch)]
pub fn paged_quantize_page_kernel(
    page_data: &mut Array<f32>,
    quant_payload: &mut Array<u32>,
    page_base_offset: u32,
    payload_base: u32,
    page_size: u32,
    kv_dim: u32,
    head_dim: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let slot_size = 2u32 * kv_dim;
        let words_per_token = (kv_dim + 3u32) / 4u32;
        let num_kv_heads = kv_dim / head_dim;
        let residual_words_per_token = 1u32;

        for token in 0..page_size {
            let slot_base = page_base_offset + token * slot_size;
            let key_payload_base = payload_base + token * words_per_token;
            let value_payload_base =
                payload_base + page_size * words_per_token + token * words_per_token;

            for word in 0..words_per_token {
                let mut packed_key = 0u32;
                let mut packed_value = 0u32;

                for byte_idx in 0..4u32 {
                    let channel = word * 4u32 + byte_idx;
                    if channel < kv_dim {
                        let mut key_min = page_data[page_base_offset + channel];
                        let mut key_max = key_min;
                        let mut value_min = page_data[page_base_offset + kv_dim + channel];
                        let mut value_max = value_min;

                        for scan_token in 1..page_size {
                            let scan_base = page_base_offset + scan_token * slot_size;
                            let key = page_data[scan_base + channel];
                            let value = page_data[scan_base + kv_dim + channel];
                            key_min = key_min.min(key);
                            key_max = key_max.max(key);
                            value_min = value_min.min(value);
                            value_max = value_max.max(value);
                        }

                        let mut key_scale = 1.0f32;
                        if key_max > key_min {
                            key_scale = (key_max - key_min) / 255.0f32;
                        }
                        let mut value_scale = 1.0f32;
                        if value_max > value_min {
                            value_scale = (value_max - value_min) / 255.0f32;
                        }

                        let key = page_data[slot_base + channel];
                        let value = page_data[slot_base + kv_dim + channel];

                        let mut quant_key = (key - key_min) / key_scale + 0.5f32;
                        if quant_key < 0.0f32 {
                            quant_key = 0.0f32;
                        }
                        if quant_key > 255.0f32 {
                            quant_key = 255.0f32;
                        }

                        let mut quant_value = (value - value_min) / value_scale + 0.5f32;
                        if quant_value < 0.0f32 {
                            quant_value = 0.0f32;
                        }
                        if quant_value > 255.0f32 {
                            quant_value = 255.0f32;
                        }

                        packed_key |= (quant_key as u32) << (byte_idx * 8u32);
                        packed_value |= (quant_value as u32) << (byte_idx * 8u32);
                    }
                }

                quant_payload[key_payload_base + word] = packed_key;
                quant_payload[value_payload_base + word] = packed_value;
            }

            let residual_word_base = payload_base
                + 2u32 * page_size * words_per_token
                + token * num_kv_heads * residual_words_per_token;
            let residual_scale_base = page_base_offset + 4u32 * kv_dim + token * num_kv_heads;

            for kv_head in 0..num_kv_heads {
                let mut abs_sum = 0.0f32;
                let mut packed = 0u32;

                for sketch in 0..KV_QJL_SKETCH_DIM {
                    let mut projection = 0.0f32;

                    for dim in 0..head_dim {
                        let channel = kv_head * head_dim + dim;

                        let mut key_min = page_data[page_base_offset + channel];
                        let mut key_max = key_min;
                        for scan_token in 1..page_size {
                            let scan_base = page_base_offset + scan_token * slot_size;
                            let key = page_data[scan_base + channel];
                            key_min = key_min.min(key);
                            key_max = key_max.max(key);
                        }

                        let mut key_scale = 1.0f32;
                        if key_max > key_min {
                            key_scale = (key_max - key_min) / 255.0f32;
                        }

                        let key = page_data[slot_base + channel];
                        let mut quant_key = (key - key_min) / key_scale + 0.5f32;
                        if quant_key < 0.0f32 {
                            quant_key = 0.0f32;
                        }
                        if quant_key > 255.0f32 {
                            quant_key = 255.0f32;
                        }
                        let dequant_key = key_min + key_scale * quant_key;

                        let hash = (sketch + 1u32) * 1103515245u32 + (dim + 1u32) * 12345u32;
                        let mut sign = 1.0f32;
                        if (hash & 1u32) != 0u32 {
                            sign = -1.0f32;
                        }
                        projection += sign * (key - dequant_key);
                    }

                    if projection >= 0.0 {
                        packed |= 1u32 << sketch;
                    }
                    if projection < 0.0f32 {
                        abs_sum += -projection;
                    } else {
                        abs_sum += projection;
                    }
                }

                quant_payload[residual_word_base + kv_head] = packed;
                page_data[residual_scale_base + kv_head] = abs_sum / KV_QJL_SKETCH_DIM_F32;
            }
        }

        let key_scale_base = page_base_offset;
        let key_zero_base = page_base_offset + kv_dim;
        let value_scale_base = page_base_offset + 2u32 * kv_dim;
        let value_zero_base = page_base_offset + 3u32 * kv_dim;

        for channel in 0..kv_dim {
            let mut key_min = page_data[page_base_offset + channel];
            let mut key_max = key_min;
            let mut value_min = page_data[page_base_offset + kv_dim + channel];
            let mut value_max = value_min;

            for token in 1..page_size {
                let slot_base = page_base_offset + token * slot_size;
                let key = page_data[slot_base + channel];
                let value = page_data[slot_base + kv_dim + channel];
                key_min = key_min.min(key);
                key_max = key_max.max(key);
                value_min = value_min.min(value);
                value_max = value_max.max(value);
            }

            let mut key_scale = 1.0f32;
            if key_max > key_min {
                key_scale = (key_max - key_min) / 255.0f32;
            }
            let mut value_scale = 1.0f32;
            if value_max > value_min {
                value_scale = (value_max - value_min) / 255.0f32;
            }

            page_data[key_scale_base + channel] = key_scale;
            page_data[key_zero_base + channel] = key_min;
            page_data[value_scale_base + channel] = value_scale;
            page_data[value_zero_base + channel] = value_min;
        }
    }
}

/// Append a K or V vector into one slot of a paged KV cache.
///
/// `page_base_offset` is the page start in the flat page-pool buffer.
/// `value_offset` is `0` for K writes and `kv_dim` for V writes.
#[cube(launch)]
pub fn paged_cache_append_kernel(
    new_kv: &Array<f32>,
    page_data: &mut Array<f32>,
    page_base_offset: u32,
    offset_in_page: u32,
    kv_dim: u32,
    value_offset: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let slot_size = 2u32 * kv_dim;
        let dst_base = page_base_offset + offset_in_page * slot_size + value_offset;
        for i in 0..kv_dim {
            page_data[dst_base + i] = new_kv[i];
        }
    }
}

/// Compute attention scores against a paged key cache in flat GPU memory.
#[cube(launch)]
pub fn paged_attention_scores_kernel(
    q_head: &Array<f32>,
    page_data: &Array<f32>,
    quant_payload: &Array<u32>,
    page_table: &Array<u32>,
    scores: &mut Array<f32>,
    page_stride: u32,
    page_size: u32,
    num_tokens: u32,
    kv_head: u32,
    head_dim: u32,
    scale: f32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let kv_dim = page_stride / (2u32 * page_size);
        let slot_size = 2u32 * kv_dim;
        let words_per_token = (kv_dim + 3u32) / 4u32;
        let num_kv_heads = kv_dim / head_dim;
        let residual_words_per_token = 1u32;
        let meta_stride = 4u32 * kv_dim + page_size * num_kv_heads;
        let payload_words_per_page = 2u32 * page_size * words_per_token
            + page_size * num_kv_heads * residual_words_per_token;
        for pos in 0..num_tokens {
            let logical_page = pos / page_size;
            let offset_in_page = pos % page_size;
            let encoded_page = page_table[logical_page];
            let page_id = encoded_page & 0x7FFF_FFFFu32;
            let page_base = page_id * page_stride;

            let mut dot = 0.0f32;
            if (encoded_page & 0x8000_0000u32) != 0u32 {
                let meta_base = page_base;
                let key_scale_base = meta_base + kv_head * head_dim;
                let key_zero_base = meta_base + kv_dim + kv_head * head_dim;
                let residual_scale_base = meta_base + 4u32 * kv_dim + offset_in_page * num_kv_heads;
                let payload_base = page_id * payload_words_per_page;
                let key_payload_base = payload_base + offset_in_page * words_per_token;
                let residual_payload_base =
                    payload_base + 2u32 * page_size * words_per_token + offset_in_page * num_kv_heads * residual_words_per_token;
                for dim in 0..head_dim {
                    let channel = kv_head * head_dim + dim;
                    let word_idx = key_payload_base + channel / 4u32;
                    let shift = (channel % 4u32) * 8u32;
                    let quant = ((quant_payload[word_idx] >> shift) & 0xFFu32) as f32;
                    let zero = page_data[key_zero_base + dim];
                    let channel_scale = page_data[key_scale_base + dim];
                    let dequant = zero + channel_scale * quant;
                    dot += q_head[dim] * dequant;
                }

                if residual_words_per_token != 0u32 {
                    let residual_scale_idx = residual_scale_base + kv_head;
                    let mut correction = 0.0f32;
                    let residual_scale = page_data[residual_scale_idx];
                    for sketch in 0..KV_QJL_SKETCH_DIM {
                        let word_idx =
                            residual_payload_base + kv_head * residual_words_per_token + sketch / 32u32;
                        let bit = (quant_payload[word_idx] >> (sketch % 32u32)) & 1u32;
                        let mut sketch_sign = 1.0f32;
                        if bit == 0u32 {
                            sketch_sign = -1.0f32;
                        }
                        let mut proj_q = 0.0f32;
                        for dim in 0..head_dim {
                            let hash = (sketch + 1u32) * 1103515245u32 + (dim + 1u32) * 12345u32;
                            let mut sign = 1.0f32;
                            if (hash & 1u32) != 0u32 {
                                sign = -1.0f32;
                            }
                            proj_q += sign * q_head[dim];
                        }
                        correction += sketch_sign * proj_q;
                    }
                    dot += residual_scale * correction / KV_QJL_SKETCH_DIM_F32;
                }
            } else {
                let k_base = page_base + offset_in_page * slot_size + kv_head * head_dim;
                for dim in 0..head_dim {
                    dot += q_head[dim] * page_data[k_base + dim];
                }
            }
            scores[pos] = dot * scale;
        }
    }
}

/// Contract attention weights against a paged value cache in flat GPU memory.
#[cube(launch)]
pub fn paged_attention_contract_kernel(
    scores: &Array<f32>,
    page_data: &Array<f32>,
    quant_payload: &Array<u32>,
    page_table: &Array<u32>,
    output: &mut Array<f32>,
    page_stride: u32,
    page_size: u32,
    num_tokens: u32,
    kv_head: u32,
    head_dim: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let kv_dim = page_stride / (2u32 * page_size);
        let slot_size = 2u32 * kv_dim;
        let words_per_token = (kv_dim + 3u32) / 4u32;
        let num_kv_heads = kv_dim / head_dim;
        let residual_words_per_token = 1u32;
        let payload_words_per_page = 2u32 * page_size * words_per_token
            + page_size * num_kv_heads * residual_words_per_token;
        for dim in 0..head_dim {
            let mut sum = 0.0f32;
            for pos in 0..num_tokens {
                let logical_page = pos / page_size;
                let offset_in_page = pos % page_size;
                let encoded_page = page_table[logical_page];
                let page_id = encoded_page & 0x7FFF_FFFFu32;
                if (encoded_page & 0x8000_0000u32) != 0u32 {
                    let page_base = page_id * page_stride;
                    let meta_base = page_base;
                    let value_scale_base = meta_base + 2u32 * kv_dim + kv_head * head_dim;
                    let value_zero_base = meta_base + 3u32 * kv_dim + kv_head * head_dim;
                    let channel = kv_head * head_dim + dim;
                    let quant_base = page_id * payload_words_per_page
                        + page_size * words_per_token
                        + offset_in_page * words_per_token;
                    let word_idx = quant_base + channel / 4u32;
                    let shift = (channel % 4u32) * 8u32;
                    let quant = ((quant_payload[word_idx] >> shift) & 0xFFu32) as f32;
                    let dequant = page_data[value_zero_base + dim] + page_data[value_scale_base + dim] * quant;
                    sum += scores[pos] * dequant;
                } else {
                    let page_base = page_id * page_stride;
                    let v_base =
                        page_base + offset_in_page * slot_size + kv_dim + kv_head * head_dim + dim;
                    sum += scores[pos] * page_data[v_base];
                }
            }
            output[dim] = sum;
        }
    }
}

/// Copy one complete page from contiguous K/V cache buffers into paged storage.
#[cube(launch)]
pub fn contiguous_to_paged_page_kernel(
    keys: &Array<f32>,
    values: &Array<f32>,
    page_data: &mut Array<f32>,
    page_base_offset: u32,
    src_token_offset: u32,
    page_size: u32,
    kv_dim: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let slot_size = 2u32 * kv_dim;
        for token in 0..page_size {
            let src_base = (src_token_offset + token) * kv_dim;
            let dst_base = page_base_offset + token * slot_size;
            for dim in 0..kv_dim {
                page_data[dst_base + dim] = keys[src_base + dim];
                page_data[dst_base + kv_dim + dim] = values[src_base + dim];
            }
        }
    }
}

/// Copy one complete page from paged storage back into contiguous K/V caches.
#[cube(launch)]
pub fn paged_to_contiguous_page_kernel(
    page_data: &Array<f32>,
    keys: &mut Array<f32>,
    values: &mut Array<f32>,
    page_base_offset: u32,
    dst_token_offset: u32,
    page_size: u32,
    kv_dim: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let slot_size = 2u32 * kv_dim;
        for token in 0..page_size {
            let src_base = page_base_offset + token * slot_size;
            let dst_base = (dst_token_offset + token) * kv_dim;
            for dim in 0..kv_dim {
                keys[dst_base + dim] = page_data[src_base + dim];
                values[dst_base + dim] = page_data[src_base + kv_dim + dim];
            }
        }
    }
}
