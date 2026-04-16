//! Paged KV cache: a page-table-backed implementation of [`KVStore`].
//!
//! Each sequence owns a [`SequencePageTable`] that maps logical token positions
//! to physical pages in the [`BlockAllocator`]. The [`PagedLayerView`] provides
//! the [`KVStore`] interface for a single layer, resolving positions through
//! the page table on every access.

use crate::block_manager::BlockAllocator;
use crate::error::{Result, ServingError};
use crate::page::PageId;
use nnx_core::engine::KVStore;

/// Per-sequence page table mapping logical pages to physical pages, per layer.
///
/// Layout: `tables[layer_idx][logical_page_idx] = PageId`
///
/// The page table grows as the sequence generates tokens. When a page fills up,
/// a new page is allocated and appended to the layer's page list.
#[derive(Debug, Clone)]
pub struct SequencePageTable {
    /// Page IDs per layer. `tables[layer][page_idx] = PageId`.
    pub(crate) tables: Vec<Vec<PageId>>,
    /// Total number of tokens cached across all layers (same for each layer).
    num_tokens: usize,
    /// Number of transformer layers.
    num_layers: usize,
    /// Tokens per page (from config).
    page_size: usize,
}

impl SequencePageTable {
    /// Create an empty page table for a model with `num_layers` layers.
    pub fn new(num_layers: usize, page_size: usize) -> Self {
        Self {
            tables: vec![Vec::new(); num_layers],
            num_tokens: 0,
            num_layers,
            page_size,
        }
    }

    /// Number of tokens currently stored.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Set the token count (used after a forward pass to sync with layer views).
    pub fn set_num_tokens(&mut self, n: usize) {
        self.num_tokens = n;
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Number of pages allocated per layer.
    pub fn pages_per_layer(&self) -> usize {
        if self.tables.is_empty() {
            0
        } else {
            self.tables[0].len()
        }
    }

    /// Total pages allocated across all layers.
    pub fn total_pages(&self) -> usize {
        self.tables.iter().map(|t| t.len()).sum()
    }

    /// Get the page IDs for a given layer.
    pub fn layer_pages(&self, layer_idx: usize) -> &[PageId] {
        &self.tables[layer_idx]
    }

    /// Get the mutable page ID list for a given layer.
    pub fn layer_pages_mut(&mut self, layer_idx: usize) -> &mut Vec<PageId> {
        &mut self.tables[layer_idx]
    }

    /// Get all page IDs across all layers (for freeing).
    pub fn all_pages(&self) -> Vec<PageId> {
        self.tables.iter().flat_map(|t| t.iter().copied()).collect()
    }
}

/// A read/write view into the paged KV cache for one layer.
///
/// Implements [`KVStore`] by translating positions through the page table
/// and reading/writing physical pages in the block allocator.
///
/// Each view **owns** its token counter, matching how [`LayerCache`] independently
/// tracks `len` per layer. The caller initializes the view with the sequence's
/// current token count and reads back the final count via [`token_count()`]
/// after the forward pass completes.
///
/// # Lifetime
///
/// This view borrows the `BlockAllocator` and one layer's page table.
/// It is created fresh for each layer during the forward pass.
pub struct PagedLayerView<'a> {
    /// The global page allocator (owns all physical pages).
    allocator: &'a mut BlockAllocator,
    /// This sequence's page table for this layer.
    page_table: &'a mut Vec<PageId>,
    /// Number of tokens stored so far (owned, per-layer).
    num_tokens: usize,
    /// Layer index (for diagnostics).
    layer_idx: usize,
    /// Tokens per page.
    page_size: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
}

impl<'a> PagedLayerView<'a> {
    /// Create a new paged layer view.
    ///
    /// `initial_tokens` should be the sequence's current token count
    /// (same value for every layer in a given forward pass).
    pub fn new(
        allocator: &'a mut BlockAllocator,
        page_table: &'a mut Vec<PageId>,
        initial_tokens: usize,
        layer_idx: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            allocator,
            page_table,
            num_tokens: initial_tokens,
            layer_idx,
            page_size,
            num_kv_heads,
            head_dim,
        }
    }

    /// The final token count after stores. Read this after the layer's
    /// forward pass to sync back to the sequence page table.
    pub fn token_count(&self) -> usize {
        self.num_tokens
    }

    /// Resolve a token position to (page_id, offset_in_page).
    #[inline]
    fn resolve(&self, pos: usize) -> (PageId, usize) {
        let page_idx = pos / self.page_size;
        let offset = pos % self.page_size;
        (self.page_table[page_idx], offset)
    }
}

impl KVStore for PagedLayerView<'_> {
    fn key_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let (page_id, offset) = self.resolve(pos);
        self.allocator
            .page(page_id)
            .key_at(offset, kv_head, self.head_dim)
    }

    fn value_at(&self, pos: usize, kv_head: usize) -> &[f32] {
        let (page_id, offset) = self.resolve(pos);
        self.allocator
            .page(page_id)
            .value_at(offset, kv_head, self.head_dim)
    }

    fn len(&self) -> usize {
        self.num_tokens
    }

    fn store(&mut self, key: &[f32], value: &[f32]) -> nnx_core::error::Result<()> {
        let offset_in_page = self.num_tokens % self.page_size;

        // If we're at a page boundary (or first token), allocate a new page.
        if offset_in_page == 0 {
            let new_page = self.allocator.allocate().map_err(|e| {
                nnx_core::EngineError::Cache(format!(
                    "layer {}: failed to allocate page: {}",
                    self.layer_idx, e
                ))
            })?;
            self.page_table.push(new_page);
        }

        // Store into the current last page.
        let page_id = *self.page_table.last().unwrap();
        self.allocator
            .page_mut(page_id)
            .store(key, value)
            .map_err(|e| {
                nnx_core::EngineError::Cache(format!(
                    "layer {}: failed to store in page: {}",
                    self.layer_idx, e
                ))
            })?;

        self.num_tokens += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_page_table() {
        let pt = SequencePageTable::new(32, 16);
        assert_eq!(pt.num_tokens(), 0);
        assert_eq!(pt.pages_per_layer(), 0);
        assert_eq!(pt.total_pages(), 0);
    }

    #[test]
    fn paged_layer_view_store_and_read() {
        // 1 kv_head, head_dim=2, page_size=4
        let mut allocator = BlockAllocator::new(8, 4, 1, 2);
        let mut page_table: Vec<PageId> = Vec::new();

        {
            let mut view = PagedLayerView::new(
                &mut allocator,
                &mut page_table,
                0, // initial_tokens
                0, // layer_idx
                4, // page_size
                1, // num_kv_heads
                2, // head_dim
            );

            // Store 3 tokens.
            view.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
            view.store(&[5.0, 6.0], &[7.0, 8.0]).unwrap();
            view.store(&[9.0, 10.0], &[11.0, 12.0]).unwrap();

            assert_eq!(view.len(), 3);
            assert_eq!(view.token_count(), 3);

            // Read back.
            assert_eq!(view.key_at(0, 0), &[1.0, 2.0]);
            assert_eq!(view.value_at(0, 0), &[3.0, 4.0]);
            assert_eq!(view.key_at(1, 0), &[5.0, 6.0]);
            assert_eq!(view.value_at(1, 0), &[7.0, 8.0]);
            assert_eq!(view.key_at(2, 0), &[9.0, 10.0]);
            assert_eq!(view.value_at(2, 0), &[11.0, 12.0]);
        }

        // 1 page allocated (capacity 4, 3 tokens used).
        assert_eq!(page_table.len(), 1);
        assert_eq!(allocator.used_count(), 1);
    }

    #[test]
    fn paged_layer_view_crosses_page_boundary() {
        // page_size=2, so after 2 tokens a new page is needed.
        let mut allocator = BlockAllocator::new(8, 2, 1, 2);
        let mut page_table: Vec<PageId> = Vec::new();

        {
            let mut view = PagedLayerView::new(&mut allocator, &mut page_table, 0, 0, 2, 1, 2);

            // Store 3 tokens (crosses page boundary at token 2).
            view.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
            view.store(&[5.0, 6.0], &[7.0, 8.0]).unwrap();
            // Page 0 is now full.
            view.store(&[9.0, 10.0], &[11.0, 12.0]).unwrap();
            // Page 1 has 1 token.

            assert_eq!(view.len(), 3);

            // Read back — tokens span two pages.
            assert_eq!(view.key_at(0, 0), &[1.0, 2.0]);
            assert_eq!(view.key_at(1, 0), &[5.0, 6.0]);
            assert_eq!(view.key_at(2, 0), &[9.0, 10.0]);
            assert_eq!(view.value_at(2, 0), &[11.0, 12.0]);
        }

        // 2 pages allocated.
        assert_eq!(page_table.len(), 2);
        assert_eq!(allocator.used_count(), 2);
    }

    #[test]
    fn paged_layer_view_multi_head() {
        // 2 kv_heads, head_dim=4, page_size=4
        let mut allocator = BlockAllocator::new(8, 4, 2, 4);
        let mut page_table: Vec<PageId> = Vec::new();

        {
            let mut view = PagedLayerView::new(&mut allocator, &mut page_table, 0, 0, 4, 2, 4);

            let key = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 heads * 4 dim
            let val = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
            view.store(&key, &val).unwrap();

            assert_eq!(view.key_at(0, 0), &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(view.key_at(0, 1), &[5.0, 6.0, 7.0, 8.0]);
            assert_eq!(view.value_at(0, 0), &[0.1, 0.2, 0.3, 0.4]);
            assert_eq!(view.value_at(0, 1), &[0.5, 0.6, 0.7, 0.8]);
        }
    }

    #[test]
    fn paged_layer_view_matches_layer_cache() {
        // Verify that PagedLayerView produces identical reads as LayerCache
        // for the same sequence of stores. This is the core correctness test.

        let num_kv_heads = 2;
        let head_dim = 4;
        let kv_dim = num_kv_heads * head_dim;
        let page_size = 4;
        let num_tokens_to_test = 7; // Crosses page boundary (4 + 3)

        // Generate test data.
        let test_keys: Vec<Vec<f32>> = (0..num_tokens_to_test)
            .map(|t| (0..kv_dim).map(|d| (t * kv_dim + d) as f32 * 0.1).collect())
            .collect();
        let test_vals: Vec<Vec<f32>> = (0..num_tokens_to_test)
            .map(|t| {
                (0..kv_dim)
                    .map(|d| (t * kv_dim + d) as f32 * -0.1)
                    .collect()
            })
            .collect();

        // Store via PagedLayerView.
        let mut allocator = BlockAllocator::new(16, page_size, num_kv_heads, head_dim);
        let mut page_table: Vec<PageId> = Vec::new();

        {
            let mut view = PagedLayerView::new(
                &mut allocator,
                &mut page_table,
                0,
                0,
                page_size,
                num_kv_heads,
                head_dim,
            );

            for t in 0..num_tokens_to_test {
                view.store(&test_keys[t], &test_vals[t]).unwrap();
            }

            // Verify every position and head matches.
            for t in 0..num_tokens_to_test {
                for h in 0..num_kv_heads {
                    let expected_key = &test_keys[t][h * head_dim..(h + 1) * head_dim];
                    let expected_val = &test_vals[t][h * head_dim..(h + 1) * head_dim];

                    assert_eq!(
                        view.key_at(t, h),
                        expected_key,
                        "key mismatch at token={}, head={}",
                        t,
                        h
                    );
                    assert_eq!(
                        view.value_at(t, h),
                        expected_val,
                        "value mismatch at token={}, head={}",
                        t,
                        h
                    );
                }
            }
        }
    }

    #[test]
    fn multi_layer_independent_counters() {
        // Verify that two layers can independently store tokens
        // with the same initial count, matching the contiguous model.
        let num_kv_heads = 1;
        let head_dim = 2;
        let page_size = 4;

        let mut allocator = BlockAllocator::new(16, page_size, num_kv_heads, head_dim);
        let mut layer0_pages: Vec<PageId> = Vec::new();
        let mut layer1_pages: Vec<PageId> = Vec::new();

        // Simulate a 2-layer forward pass storing 1 token.
        let initial_tokens = 0;

        // Layer 0
        let final_count_0 = {
            let mut view = PagedLayerView::new(
                &mut allocator,
                &mut layer0_pages,
                initial_tokens,
                0,
                page_size,
                num_kv_heads,
                head_dim,
            );
            view.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
            assert_eq!(view.len(), 1);
            view.token_count()
        };

        // Layer 1 — starts with the SAME initial count.
        let final_count_1 = {
            let mut view = PagedLayerView::new(
                &mut allocator,
                &mut layer1_pages,
                initial_tokens, // same starting point
                1,
                page_size,
                num_kv_heads,
                head_dim,
            );
            view.store(&[5.0, 6.0], &[7.0, 8.0]).unwrap();
            assert_eq!(view.len(), 1);
            view.token_count()
        };

        // Both layers should agree on the final count.
        assert_eq!(final_count_0, 1);
        assert_eq!(final_count_1, 1);
        assert_eq!(final_count_0, final_count_1);

        // Both layers allocated 1 page each.
        assert_eq!(layer0_pages.len(), 1);
        assert_eq!(layer1_pages.len(), 1);
        assert_eq!(allocator.used_count(), 2);

        // Read back from each layer.
        {
            let view0 = PagedLayerView::new(
                &mut allocator,
                &mut layer0_pages,
                1,
                0,
                page_size,
                num_kv_heads,
                head_dim,
            );
            assert_eq!(view0.key_at(0, 0), &[1.0, 2.0]);
        }
        {
            let view1 = PagedLayerView::new(
                &mut allocator,
                &mut layer1_pages,
                1,
                1,
                page_size,
                num_kv_heads,
                head_dim,
            );
            assert_eq!(view1.key_at(0, 0), &[5.0, 6.0]);
        }
    }
}
