//! GPU-resident paged KV cache storage and paged attention kernels.
//!
//! The backing storage is a single flat GPU buffer where page `i` starts at:
//! `i * page_size * 2 * num_kv_heads * head_dim`.
//! Each page preserves the CPU paged-cache layout:
//! `[token][K|V][kv_head][head_dim]`.

use cubecl::prelude::*;
use nnx_core::PageId;
use nnx_core::backend::KernelBackend;

use crate::backend::{CubeclBackend, GpuBuffer};

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
        let kv_dim = num_kv_heads * head_dim;
        let page_stride = page_size * 2 * kv_dim;
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

        Self {
            flat_buffer,
            pages,
            page_size,
            num_kv_heads,
            head_dim,
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
    page_table: &Array<u32>,
    scores: &mut Array<f32>,
    page_stride: u32,
    page_size: u32,
    num_tokens: u32,
    kv_head: u32,
    head_dim: u32,
    kv_dim: u32,
    scale: f32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let slot_size = 2u32 * kv_dim;
        for pos in 0..num_tokens {
            let logical_page = pos / page_size;
            let offset_in_page = pos % page_size;
            let page_id = page_table[logical_page];
            let page_base = page_id * page_stride;
            let k_base = page_base + offset_in_page * slot_size + kv_head * head_dim;

            let mut dot = 0.0f32;
            for dim in 0..head_dim {
                dot += q_head[dim] * page_data[k_base + dim];
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
    page_table: &Array<u32>,
    output: &mut Array<f32>,
    page_stride: u32,
    page_size: u32,
    num_tokens: u32,
    kv_head: u32,
    head_dim: u32,
    kv_dim: u32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let slot_size = 2u32 * kv_dim;
        for dim in 0..head_dim {
            let mut sum = 0.0f32;
            for pos in 0..num_tokens {
                let logical_page = pos / page_size;
                let offset_in_page = pos % page_size;
                let page_id = page_table[logical_page];
                let page_base = page_id * page_stride;
                let v_base =
                    page_base + offset_in_page * slot_size + kv_dim + kv_head * head_dim + dim;
                sum += scores[pos] * page_data[v_base];
            }
            output[dim] = sum;
        }
    }
}
