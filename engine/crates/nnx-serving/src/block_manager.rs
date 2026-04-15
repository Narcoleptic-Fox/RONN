//! Block allocator for paged KV cache.
//!
//! Manages a fixed pool of [`PhysicalPage`]s with reference counting and
//! copy-on-write support. One allocator per loaded model — all sequences
//! sharing that model draw from the same page pool.

use crate::error::{Result, ServingError};
use crate::page::{PageId, PhysicalPage};

/// A fixed-pool allocator for KV cache pages.
///
/// Pre-allocates all pages at startup. Allocation and deallocation are O(1).
/// Reference counting enables copy-on-write for prefix sharing and beam search.
///
/// # Thread Safety
///
/// The allocator is designed to be mutated only by the scheduler (between
/// forward passes). During forward passes, pages are read concurrently by
/// attention kernels via shared references. No locking is needed because
/// mutations and reads do not overlap.
#[derive(Debug)]
pub struct BlockAllocator {
    /// All physical pages, indexed by PageId.
    pages: Vec<PhysicalPage>,
    /// Free list (stack — LIFO for cache locality).
    free_list: Vec<PageId>,
    /// Reference count per page.
    ref_counts: Vec<u32>,
    /// Model geometry (for page creation).
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
}

/// Read-only statistics about the allocator state.
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    /// Total number of pages in the pool.
    pub total_pages: usize,
    /// Number of currently free pages.
    pub free_pages: usize,
    /// Number of pages currently in use.
    pub used_pages: usize,
    /// Memory per page in bytes.
    pub page_memory_bytes: usize,
    /// Total pool memory in bytes.
    pub total_memory_bytes: usize,
    /// Total memory currently in use.
    pub used_memory_bytes: usize,
}

impl BlockAllocator {
    /// Create a new allocator with a fixed number of pages.
    ///
    /// All pages are pre-allocated and placed on the free list.
    pub fn new(
        max_pages: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let pages: Vec<PhysicalPage> = (0..max_pages)
            .map(|_| PhysicalPage::new(page_size, num_kv_heads, head_dim))
            .collect();

        // All pages start free (reversed so PageId(0) is popped first).
        let free_list: Vec<PageId> = (0..max_pages as u32).rev().map(PageId).collect();
        let ref_counts = vec![0u32; max_pages];

        Self {
            pages,
            free_list,
            ref_counts,
            num_kv_heads,
            head_dim,
            page_size,
        }
    }

    /// Allocate a single page. Returns `OutOfPages` if the pool is exhausted.
    pub fn allocate(&mut self) -> Result<PageId> {
        let page_id = self.free_list.pop().ok_or_else(|| {
            ServingError::OutOfPages(format!(
                "all {} pages in use",
                self.pages.len()
            ))
        })?;
        self.ref_counts[page_id.0 as usize] = 1;
        self.pages[page_id.0 as usize].reset();
        Ok(page_id)
    }

    /// Increment the reference count for a page (for prefix sharing / beam fork).
    pub fn inc_ref(&mut self, page_id: PageId) -> Result<()> {
        let idx = page_id.0 as usize;
        if idx >= self.ref_counts.len() {
            return Err(ServingError::InvalidPage(format!(
                "page id {} out of range (pool size {})",
                idx,
                self.ref_counts.len()
            )));
        }
        if self.ref_counts[idx] == 0 {
            return Err(ServingError::RefCount(format!(
                "cannot inc_ref on free page {}",
                idx
            )));
        }
        self.ref_counts[idx] = self.ref_counts[idx].checked_add(1).ok_or_else(|| {
            ServingError::RefCount(format!("ref count overflow on page {}", idx))
        })?;
        Ok(())
    }

    /// Decrement the reference count. If it reaches zero, the page is freed.
    pub fn dec_ref(&mut self, page_id: PageId) -> Result<()> {
        let idx = page_id.0 as usize;
        if idx >= self.ref_counts.len() {
            return Err(ServingError::InvalidPage(format!(
                "page id {} out of range (pool size {})",
                idx,
                self.ref_counts.len()
            )));
        }
        if self.ref_counts[idx] == 0 {
            return Err(ServingError::RefCount(format!(
                "double-free on page {}",
                idx
            )));
        }
        self.ref_counts[idx] -= 1;
        if self.ref_counts[idx] == 0 {
            self.free_list.push(page_id);
        }
        Ok(())
    }

    /// Get the reference count for a page.
    pub fn ref_count(&self, page_id: PageId) -> u32 {
        self.ref_counts[page_id.0 as usize]
    }

    /// Copy-on-write: if the page has ref_count > 1, allocate a new page,
    /// copy the data, decrement the old page's ref_count, and return the new page.
    /// If ref_count == 1, return the same page (no copy needed).
    pub fn cow_if_shared(&mut self, page_id: PageId) -> Result<PageId> {
        let idx = page_id.0 as usize;
        if self.ref_counts[idx] == 0 {
            return Err(ServingError::CopyOnWrite(format!(
                "cannot CoW free page {}",
                idx
            )));
        }
        if self.ref_counts[idx] == 1 {
            // Exclusively owned — no copy needed.
            return Ok(page_id);
        }

        // Shared — need to copy.
        let new_id = self.allocate()?;
        let new_idx = new_id.0 as usize;

        // Copy page data. We need to use split borrowing to avoid aliasing.
        // Since page_id != new_id (allocate picks from free list, old page is in use),
        // we can safely split the slice.
        let (src, dst) = if idx < new_idx {
            let (left, right) = self.pages.split_at_mut(new_idx);
            (&left[idx], &mut right[0])
        } else {
            let (left, right) = self.pages.split_at_mut(idx);
            (&right[0], &mut left[new_idx])
        };
        dst.copy_from(src);

        // Decrement old page's ref count (it's still shared by others).
        self.ref_counts[idx] -= 1;

        Ok(new_id)
    }

    /// Read-only access to a physical page.
    #[inline]
    pub fn page(&self, page_id: PageId) -> &PhysicalPage {
        &self.pages[page_id.0 as usize]
    }

    /// Mutable access to a physical page. Caller must ensure ref_count == 1.
    #[inline]
    pub fn page_mut(&mut self, page_id: PageId) -> &mut PhysicalPage {
        debug_assert_eq!(
            self.ref_counts[page_id.0 as usize], 1,
            "mutable access to shared page {} (ref_count={})",
            page_id.0,
            self.ref_counts[page_id.0 as usize]
        );
        &mut self.pages[page_id.0 as usize]
    }

    /// Number of free pages remaining.
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// Total number of pages in the pool.
    pub fn total_count(&self) -> usize {
        self.pages.len()
    }

    /// Number of pages currently in use.
    pub fn used_count(&self) -> usize {
        self.total_count() - self.free_count()
    }

    /// Page size in tokens.
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

    /// Get allocator statistics.
    pub fn stats(&self) -> AllocatorStats {
        let page_mem = if self.pages.is_empty() {
            0
        } else {
            self.pages[0].memory_bytes()
        };
        AllocatorStats {
            total_pages: self.total_count(),
            free_pages: self.free_count(),
            used_pages: self.used_count(),
            page_memory_bytes: page_mem,
            total_memory_bytes: page_mem * self.total_count(),
            used_memory_bytes: page_mem * self.used_count(),
        }
    }

    /// Free all pages owned by a sequence (given its page table).
    /// Decrements ref counts; pages with ref_count reaching 0 are freed.
    pub fn free_sequence_pages(&mut self, page_ids: &[PageId]) -> Result<()> {
        for &pid in page_ids {
            if pid != PageId::INVALID {
                self.dec_ref(pid)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_allocator(max_pages: usize) -> BlockAllocator {
        BlockAllocator::new(max_pages, 4, 2, 4) // 4 tokens/page, 2 kv_heads, head_dim=4
    }

    #[test]
    fn allocate_and_free() {
        let mut alloc = make_allocator(8);
        assert_eq!(alloc.free_count(), 8);

        let p0 = alloc.allocate().unwrap();
        assert_eq!(alloc.free_count(), 7);
        assert_eq!(alloc.ref_count(p0), 1);

        alloc.dec_ref(p0).unwrap();
        assert_eq!(alloc.free_count(), 8);
        assert_eq!(alloc.ref_count(p0), 0);
    }

    #[test]
    fn exhaustion_returns_error() {
        let mut alloc = make_allocator(2);
        alloc.allocate().unwrap();
        alloc.allocate().unwrap();
        assert!(alloc.allocate().is_err());
    }

    #[test]
    fn double_free_returns_error() {
        let mut alloc = make_allocator(4);
        let p0 = alloc.allocate().unwrap();
        alloc.dec_ref(p0).unwrap();
        assert!(alloc.dec_ref(p0).is_err());
    }

    #[test]
    fn inc_ref_on_free_page_returns_error() {
        let mut alloc = make_allocator(4);
        let p0 = alloc.allocate().unwrap();
        alloc.dec_ref(p0).unwrap();
        assert!(alloc.inc_ref(p0).is_err());
    }

    #[test]
    fn ref_counting() {
        let mut alloc = make_allocator(4);
        let p0 = alloc.allocate().unwrap();
        assert_eq!(alloc.ref_count(p0), 1);

        alloc.inc_ref(p0).unwrap();
        assert_eq!(alloc.ref_count(p0), 2);

        alloc.dec_ref(p0).unwrap();
        assert_eq!(alloc.ref_count(p0), 1);
        assert_eq!(alloc.free_count(), 3); // still in use

        alloc.dec_ref(p0).unwrap();
        assert_eq!(alloc.ref_count(p0), 0);
        assert_eq!(alloc.free_count(), 4); // freed
    }

    #[test]
    fn cow_exclusive_returns_same_page() {
        let mut alloc = make_allocator(4);
        let p0 = alloc.allocate().unwrap();
        assert_eq!(alloc.ref_count(p0), 1);

        let result = alloc.cow_if_shared(p0).unwrap();
        assert_eq!(result, p0); // no copy, same page
        assert_eq!(alloc.used_count(), 1);
    }

    #[test]
    fn cow_shared_copies_data() {
        let mut alloc = make_allocator(4);
        let p0 = alloc.allocate().unwrap();

        // Write some data.
        alloc
            .page_mut(p0)
            .store(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[0.1; 8])
            .unwrap();

        // Share the page.
        alloc.inc_ref(p0).unwrap();
        assert_eq!(alloc.ref_count(p0), 2);

        // CoW should allocate a new page and copy data.
        let p1 = alloc.cow_if_shared(p0).unwrap();
        assert_ne!(p0, p1);
        assert_eq!(alloc.ref_count(p0), 1); // old page lost one ref
        assert_eq!(alloc.ref_count(p1), 1); // new page has ref=1
        assert_eq!(alloc.used_count(), 2);

        // Data should match.
        assert_eq!(
            alloc.page(p0).key_at(0, 0, 4),
            alloc.page(p1).key_at(0, 0, 4)
        );
    }

    #[test]
    fn cow_fails_when_pool_exhausted() {
        let mut alloc = make_allocator(2);
        let p0 = alloc.allocate().unwrap();
        let _p1 = alloc.allocate().unwrap();

        alloc.inc_ref(p0).unwrap(); // share p0
        // Pool is full — CoW can't allocate.
        assert!(alloc.cow_if_shared(p0).is_err());
    }

    #[test]
    fn free_sequence_pages() {
        let mut alloc = make_allocator(8);
        let p0 = alloc.allocate().unwrap();
        let p1 = alloc.allocate().unwrap();
        let p2 = alloc.allocate().unwrap();
        assert_eq!(alloc.used_count(), 3);

        alloc.free_sequence_pages(&[p0, p1, p2]).unwrap();
        assert_eq!(alloc.used_count(), 0);
    }

    #[test]
    fn free_sequence_pages_skips_invalid() {
        let mut alloc = make_allocator(8);
        let p0 = alloc.allocate().unwrap();
        alloc
            .free_sequence_pages(&[p0, PageId::INVALID])
            .unwrap();
        assert_eq!(alloc.used_count(), 0);
    }

    #[test]
    fn stats_are_correct() {
        let mut alloc = make_allocator(8);
        let stats = alloc.stats();
        assert_eq!(stats.total_pages, 8);
        assert_eq!(stats.free_pages, 8);
        assert_eq!(stats.used_pages, 0);

        alloc.allocate().unwrap();
        alloc.allocate().unwrap();
        let stats = alloc.stats();
        assert_eq!(stats.free_pages, 6);
        assert_eq!(stats.used_pages, 2);
    }

    #[test]
    fn allocated_page_is_reset() {
        let mut alloc = make_allocator(2);
        let p0 = alloc.allocate().unwrap();
        alloc
            .page_mut(p0)
            .store(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[0.1; 8])
            .unwrap();
        assert_eq!(alloc.page(p0).num_filled(), 1);

        // Free and re-allocate.
        alloc.dec_ref(p0).unwrap();
        let p0_again = alloc.allocate().unwrap();
        assert_eq!(p0, p0_again); // LIFO
        assert_eq!(alloc.page(p0_again).num_filled(), 0); // reset
    }
}
