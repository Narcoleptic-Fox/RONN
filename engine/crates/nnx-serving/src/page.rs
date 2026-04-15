//! Physical page types for paged KV cache.
//!
//! Each page holds a fixed number of tokens' worth of KV data for **one layer**.
//! Layout: `[token][K|V][kv_head][head_dim]` — K and V interleaved per-token
//! for better locality during attention (both K and V are read per position).

use crate::error::{Result, ServingError};

/// Opaque identifier for a physical page in the block allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub u32);

impl PageId {
    /// Sentinel value for "no page" / "invalid".
    pub const INVALID: PageId = PageId(u32::MAX);
}

/// A physical page storing KV data for one layer.
///
/// Memory layout per token slot:
/// ```text
/// [K_head0..K_headN, V_head0..V_headN]
///  ^--- kv_dim ---^  ^--- kv_dim ---^
///  ^--------- slot_size (2 * kv_dim) ---------^
/// ```
///
/// Full page: `[slot_0, slot_1, ..., slot_{page_size-1}]`
#[derive(Debug, Clone)]
pub struct PhysicalPage {
    /// Raw f32 storage: `page_size * 2 * num_kv_heads * head_dim` elements.
    data: Vec<f32>,
    /// Number of token slots currently filled (0..=page_size).
    num_filled: usize,
    /// Tokens per page (cached from config for bounds checking).
    page_size: usize,
    /// KV dimension: `num_kv_heads * head_dim`.
    kv_dim: usize,
}

impl PhysicalPage {
    /// Create a new zeroed page.
    pub fn new(page_size: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        let total_floats = page_size * 2 * kv_dim;
        Self {
            data: vec![0.0; total_floats],
            num_filled: 0,
            page_size,
            kv_dim,
        }
    }

    /// Number of filled token slots.
    #[inline]
    pub fn num_filled(&self) -> usize {
        self.num_filled
    }

    /// Whether this page has room for more tokens.
    #[inline]
    pub fn has_room(&self) -> bool {
        self.num_filled < self.page_size
    }

    /// Whether this page is completely filled.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.num_filled == self.page_size
    }

    /// Page capacity in tokens.
    #[inline]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Size of one token slot in f32 elements: `2 * kv_dim`.
    #[inline]
    pub fn slot_size(&self) -> usize {
        2 * self.kv_dim
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 4
    }

    /// Store a token's K and V at the next available slot.
    ///
    /// `key` and `value` must each have length `kv_dim` (`num_kv_heads * head_dim`).
    pub fn store(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if self.num_filled >= self.page_size {
            return Err(ServingError::OutOfPages(format!(
                "page is full ({}/{} slots)",
                self.num_filled, self.page_size
            )));
        }
        if key.len() != self.kv_dim {
            return Err(ServingError::InvalidPage(format!(
                "key length {} != kv_dim {}",
                key.len(),
                self.kv_dim
            )));
        }
        if value.len() != self.kv_dim {
            return Err(ServingError::InvalidPage(format!(
                "value length {} != kv_dim {}",
                value.len(),
                self.kv_dim
            )));
        }

        let slot_size = self.slot_size();
        let base = self.num_filled * slot_size;
        self.data[base..base + self.kv_dim].copy_from_slice(key);
        self.data[base + self.kv_dim..base + slot_size].copy_from_slice(value);
        self.num_filled += 1;
        Ok(())
    }

    /// Read the key vector for a given token offset and KV head.
    ///
    /// Returns a slice of length `head_dim`.
    #[inline]
    pub fn key_at(&self, offset_in_page: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let base = offset_in_page * self.slot_size() + kv_head * head_dim;
        &self.data[base..base + head_dim]
    }

    /// Read the value vector for a given token offset and KV head.
    ///
    /// Returns a slice of length `head_dim`.
    #[inline]
    pub fn value_at(&self, offset_in_page: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let base = offset_in_page * self.slot_size() + self.kv_dim + kv_head * head_dim;
        &self.data[base..base + head_dim]
    }

    /// Reset this page to empty (for reuse after deallocation).
    pub fn reset(&mut self) {
        self.num_filled = 0;
        // Don't zero data — will be overwritten on next store.
    }

    /// Copy all data from another page (for CoW).
    pub fn copy_from(&mut self, other: &PhysicalPage) {
        debug_assert_eq!(self.data.len(), other.data.len());
        self.data.copy_from_slice(&other.data);
        self.num_filled = other.num_filled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_read_round_trip() {
        // 2 kv_heads, head_dim=4, page_size=4
        let mut page = PhysicalPage::new(4, 2, 4);
        let key = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 heads * 4 dim
        let val = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        page.store(&key, &val).unwrap();

        assert_eq!(page.num_filled(), 1);
        assert_eq!(page.key_at(0, 0, 4), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(page.key_at(0, 1, 4), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(page.value_at(0, 0, 4), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(page.value_at(0, 1, 4), &[0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn store_multiple_tokens() {
        let mut page = PhysicalPage::new(4, 1, 2);
        page.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        page.store(&[5.0, 6.0], &[7.0, 8.0]).unwrap();

        assert_eq!(page.num_filled(), 2);
        assert_eq!(page.key_at(0, 0, 2), &[1.0, 2.0]);
        assert_eq!(page.value_at(0, 0, 2), &[3.0, 4.0]);
        assert_eq!(page.key_at(1, 0, 2), &[5.0, 6.0]);
        assert_eq!(page.value_at(1, 0, 2), &[7.0, 8.0]);
    }

    #[test]
    fn full_page_rejects_store() {
        let mut page = PhysicalPage::new(1, 1, 2);
        page.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert!(page.is_full());

        let err = page.store(&[5.0, 6.0], &[7.0, 8.0]);
        assert!(err.is_err());
    }

    #[test]
    fn wrong_key_length_rejected() {
        let mut page = PhysicalPage::new(4, 2, 4);
        let err = page.store(&[1.0, 2.0], &[0.1; 8]); // key too short
        assert!(err.is_err());
    }

    #[test]
    fn reset_clears_fill_count() {
        let mut page = PhysicalPage::new(4, 1, 2);
        page.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(page.num_filled(), 1);
        page.reset();
        assert_eq!(page.num_filled(), 0);
        assert!(page.has_room());
    }

    #[test]
    fn copy_from_duplicates_data() {
        let mut src = PhysicalPage::new(4, 1, 2);
        src.store(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        src.store(&[5.0, 6.0], &[7.0, 8.0]).unwrap();

        let mut dst = PhysicalPage::new(4, 1, 2);
        dst.copy_from(&src);

        assert_eq!(dst.num_filled(), 2);
        assert_eq!(dst.key_at(0, 0, 2), src.key_at(0, 0, 2));
        assert_eq!(dst.key_at(1, 0, 2), src.key_at(1, 0, 2));
        assert_eq!(dst.value_at(0, 0, 2), src.value_at(0, 0, 2));
        assert_eq!(dst.value_at(1, 0, 2), src.value_at(1, 0, 2));
    }

    #[test]
    fn memory_bytes_correct() {
        // page_size=16, 8 kv_heads, head_dim=128
        // = 16 * 2 * 8 * 128 * 4 bytes = 131072
        let page = PhysicalPage::new(16, 8, 128);
        assert_eq!(page.memory_bytes(), 131_072);
    }

    #[test]
    fn has_room_and_is_full() {
        let mut page = PhysicalPage::new(2, 1, 1);
        assert!(page.has_room());
        assert!(!page.is_full());

        page.store(&[1.0], &[2.0]).unwrap();
        assert!(page.has_room());
        assert!(!page.is_full());

        page.store(&[3.0], &[4.0]).unwrap();
        assert!(!page.has_room());
        assert!(page.is_full());
    }
}
