//! Prefix caching: content-addressed page lookup with LRU eviction.
//!
//! When multiple requests share the same prompt prefix, their KV cache pages
//! can be shared instead of recomputed. This module provides the hash table
//! and eviction policy that make prefix sharing possible.
//!
//! # Design
//!
//! Pages are identified by a **chain-dependent hash** of their token content:
//!
//! ```text
//! hash(page_i) = hash(tokens_in_page_i, hash(page_{i-1}))
//! ```
//!
//! This ensures that matching an intermediate page requires matching the
//! entire prefix leading up to it, preventing false positives from
//! coincidental token overlaps in different contexts.
//!
//! Each cache entry stores the page IDs for all layers (since all layers
//! share the same token sequence). Entries are reference-counted via the
//! block allocator.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use crate::page::PageId;

/// A chain-dependent hash of a page's token content.
///
/// The hash of page `i` depends on both the tokens in page `i` and the
/// hash of page `i-1`, forming a Merkle-like chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageHash(pub u64);

impl PageHash {
    /// The initial hash for the first page (no parent).
    pub const ROOT: PageHash = PageHash(0);
}

/// Compute the chain-dependent hash for a page of tokens.
///
/// `parent_hash` is the hash of the preceding page (or `PageHash::ROOT`
/// for the first page).
pub fn page_content_hash(tokens: &[u32], parent_hash: PageHash) -> PageHash {
    let mut hasher = std::hash::DefaultHasher::new();
    parent_hash.0.hash(&mut hasher);
    for &token in tokens {
        token.hash(&mut hasher);
    }
    PageHash(hasher.finish())
}

/// Split a token sequence into page-sized chunks and compute the chain
/// of hashes. Returns `(hash, token_slice)` pairs.
pub fn compute_hash_chain(tokens: &[u32], page_size: usize) -> Vec<(PageHash, &[u32])> {
    let mut chain = Vec::new();
    let mut parent = PageHash::ROOT;

    for chunk in tokens.chunks(page_size) {
        let hash = page_content_hash(chunk, parent);
        chain.push((hash, chunk));
        parent = hash;
    }

    chain
}

/// A cached prefix entry: page IDs for all layers for one page position.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Page IDs, one per layer. `page_ids[layer_idx] = PageId`.
    page_ids: Vec<PageId>,
    /// Number of tokens in this page (always page_size for complete pages,
    /// less for the last partial page — but we only cache complete pages).
    num_tokens: usize,
}

/// Content-addressed prefix cache with LRU eviction.
///
/// Only **complete** pages (fully filled with `page_size` tokens) are cached.
/// Partial pages at the end of a prompt are not cacheable.
#[derive(Debug)]
pub struct PrefixCache {
    /// Hash → cached page entry.
    entries: HashMap<PageHash, CacheEntry>,
    /// LRU order: front = least recently used, back = most recently used.
    lru_order: VecDeque<PageHash>,
    /// Whether prefix caching is enabled.
    enabled: bool,
}

/// Result of a prefix lookup.
#[derive(Debug)]
pub struct PrefixLookup {
    /// Number of tokens that were found in the cache.
    pub cached_tokens: usize,
    /// Number of complete pages that matched.
    pub cached_pages: usize,
    /// Page IDs for the cached portion, per layer.
    /// `matched_pages[page_idx][layer_idx] = PageId`.
    pub matched_pages: Vec<Vec<PageId>>,
    /// The hash chain computed during lookup (for inserting new pages later).
    pub hash_chain: Vec<PageHash>,
}

impl PrefixCache {
    /// Create a new prefix cache.
    pub fn new(enabled: bool) -> Self {
        Self {
            entries: HashMap::new(),
            lru_order: VecDeque::new(),
            enabled,
        }
    }

    /// Look up the longest cached prefix for a token sequence.
    ///
    /// Returns information about which pages were found and their IDs.
    /// The caller should use this to skip prefill for the cached portion
    /// and share the cached pages (incrementing ref counts).
    pub fn lookup(&mut self, tokens: &[u32], page_size: usize) -> PrefixLookup {
        if !self.enabled {
            return PrefixLookup {
                cached_tokens: 0,
                cached_pages: 0,
                matched_pages: Vec::new(),
                hash_chain: Vec::new(),
            };
        }

        let chain = compute_hash_chain(tokens, page_size);
        let mut matched_pages = Vec::new();
        let mut cached_tokens = 0;

        let hash_chain: Vec<PageHash> = chain.iter().map(|(h, _)| *h).collect();

        for (hash, chunk) in &chain {
            if let Some(entry) = self.entries.get(hash) {
                matched_pages.push(entry.page_ids.clone());
                cached_tokens += entry.num_tokens;
                // Move to back of LRU (most recently used).
                self.touch(*hash);
            } else {
                // First miss breaks the chain.
                break;
            }
        }

        PrefixLookup {
            cached_tokens,
            cached_pages: matched_pages.len(),
            matched_pages,
            hash_chain,
        }
    }

    /// Check whether a hash is already in the cache.
    pub fn contains(&self, hash: &PageHash) -> bool {
        self.entries.contains_key(hash)
    }

    /// Insert a newly computed page into the cache.
    ///
    /// Returns `true` if the entry was actually inserted (new),
    /// `false` if it was a duplicate (only LRU was touched).
    ///
    /// **Callers must only increment allocator ref counts when this returns `true`.**
    /// Incrementing before calling insert risks leaking refs on duplicates.
    pub fn insert(
        &mut self,
        hash: PageHash,
        page_ids: Vec<PageId>,
        num_tokens: usize,
        page_size: usize,
    ) -> bool {
        if !self.enabled {
            return false;
        }

        // Only cache complete pages.
        if num_tokens < page_size {
            return false;
        }

        if self.entries.contains_key(&hash) {
            // Already cached — just touch for LRU, no new entry.
            self.touch(hash);
            return false;
        }

        self.entries.insert(
            hash,
            CacheEntry {
                page_ids,
                num_tokens,
            },
        );
        self.lru_order.push_back(hash);
        true
    }

    /// Evict the least recently used entry.
    ///
    /// Returns the page IDs that should have their ref counts decremented.
    /// Returns `None` if the cache is empty.
    pub fn evict_lru(&mut self) -> Option<Vec<PageId>> {
        loop {
            let hash = self.lru_order.pop_front()?;
            // The entry might have been removed already (e.g., manual removal).
            if let Some(entry) = self.entries.remove(&hash) {
                return Some(entry.page_ids);
            }
            // Otherwise keep popping stale LRU entries.
        }
    }

    /// Number of cached page entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Move a hash to the back of the LRU order (most recently used).
    fn touch(&mut self, hash: PageHash) {
        // Remove from current position (O(n) but acceptable for now).
        if let Some(pos) = self.lru_order.iter().position(|h| *h == hash) {
            self.lru_order.remove(pos);
        }
        self.lru_order.push_back(hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        let tokens = vec![1, 2, 3, 4];
        let h1 = page_content_hash(&tokens, PageHash::ROOT);
        let h2 = page_content_hash(&tokens, PageHash::ROOT);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_is_chain_dependent() {
        let tokens = vec![1, 2, 3, 4];
        let h1 = page_content_hash(&tokens, PageHash::ROOT);
        let h2 = page_content_hash(&tokens, PageHash(42));
        assert_ne!(
            h1, h2,
            "different parent hashes should produce different results"
        );
    }

    #[test]
    fn hash_differs_for_different_tokens() {
        let h1 = page_content_hash(&[1, 2, 3, 4], PageHash::ROOT);
        let h2 = page_content_hash(&[5, 6, 7, 8], PageHash::ROOT);
        assert_ne!(h1, h2);
    }

    #[test]
    fn compute_hash_chain_splits_correctly() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9]; // 9 tokens, page_size=4
        let chain = compute_hash_chain(&tokens, 4);
        assert_eq!(chain.len(), 3); // [4, 4, 1]
        assert_eq!(chain[0].1, &[1, 2, 3, 4]);
        assert_eq!(chain[1].1, &[5, 6, 7, 8]);
        assert_eq!(chain[2].1, &[9]);

        // Verify chain dependency.
        assert_ne!(chain[0].0, chain[1].0);
    }

    #[test]
    fn lookup_empty_cache_returns_zero() {
        let mut cache = PrefixCache::new(true);
        let result = cache.lookup(&[1, 2, 3, 4], 4);
        assert_eq!(result.cached_tokens, 0);
        assert_eq!(result.cached_pages, 0);
        assert!(result.matched_pages.is_empty());
    }

    #[test]
    fn lookup_disabled_cache_returns_zero() {
        let mut cache = PrefixCache::new(false);
        cache.insert(
            page_content_hash(&[1, 2, 3, 4], PageHash::ROOT),
            vec![PageId(0), PageId(1)],
            4,
            4,
        );
        let result = cache.lookup(&[1, 2, 3, 4], 4);
        assert_eq!(result.cached_tokens, 0);
    }

    #[test]
    fn insert_and_lookup_full_page() {
        let mut cache = PrefixCache::new(true);
        let page_size = 4;
        let num_layers = 2;

        let tokens = vec![1, 2, 3, 4];
        let hash = page_content_hash(&tokens, PageHash::ROOT);
        let page_ids = vec![PageId(10), PageId(11)]; // one per layer

        cache.insert(hash, page_ids.clone(), 4, page_size);
        assert_eq!(cache.len(), 1);

        let result = cache.lookup(&tokens, page_size);
        assert_eq!(result.cached_tokens, 4);
        assert_eq!(result.cached_pages, 1);
        assert_eq!(result.matched_pages[0], page_ids);
    }

    #[test]
    fn partial_page_not_cached() {
        let mut cache = PrefixCache::new(true);
        let hash = page_content_hash(&[1, 2], PageHash::ROOT);
        cache.insert(hash, vec![PageId(0)], 2, 4); // only 2 tokens, page_size=4
        assert_eq!(cache.len(), 0); // not inserted
    }

    #[test]
    fn multi_page_prefix_lookup() {
        let mut cache = PrefixCache::new(true);
        let page_size = 4;

        // Insert two pages of a prefix.
        let chain = compute_hash_chain(&[1, 2, 3, 4, 5, 6, 7, 8], page_size);
        cache.insert(chain[0].0, vec![PageId(0), PageId(1)], 4, page_size);
        cache.insert(chain[1].0, vec![PageId(2), PageId(3)], 4, page_size);

        // Look up the same prefix.
        let result = cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8], page_size);
        assert_eq!(result.cached_tokens, 8);
        assert_eq!(result.cached_pages, 2);

        // Look up a longer sequence with the same prefix.
        let result = cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], page_size);
        assert_eq!(result.cached_tokens, 8);
        assert_eq!(result.cached_pages, 2);
    }

    #[test]
    fn prefix_mismatch_breaks_chain() {
        let mut cache = PrefixCache::new(true);
        let page_size = 4;

        // Insert page 0 for tokens [1,2,3,4].
        let chain = compute_hash_chain(&[1, 2, 3, 4, 5, 6, 7, 8], page_size);
        cache.insert(chain[0].0, vec![PageId(0)], 4, page_size);
        cache.insert(chain[1].0, vec![PageId(1)], 4, page_size);

        // Look up different tokens — page 0 won't match, so page 1 is unreachable.
        let result = cache.lookup(&[9, 9, 9, 9, 5, 6, 7, 8], page_size);
        assert_eq!(result.cached_tokens, 0);
    }

    #[test]
    fn evict_lru_returns_oldest() {
        let mut cache = PrefixCache::new(true);
        let page_size = 4;

        let h1 = page_content_hash(&[1, 2, 3, 4], PageHash::ROOT);
        let h2 = page_content_hash(&[5, 6, 7, 8], PageHash::ROOT);

        cache.insert(h1, vec![PageId(0)], 4, page_size);
        cache.insert(h2, vec![PageId(1)], 4, page_size);

        // Evict — should return the first inserted (LRU).
        let evicted = cache.evict_lru().unwrap();
        assert_eq!(evicted, vec![PageId(0)]);
        assert_eq!(cache.len(), 1);

        let evicted = cache.evict_lru().unwrap();
        assert_eq!(evicted, vec![PageId(1)]);
        assert!(cache.is_empty());
    }

    #[test]
    fn lookup_touches_lru() {
        let mut cache = PrefixCache::new(true);
        let page_size = 4;

        let h1 = page_content_hash(&[1, 2, 3, 4], PageHash::ROOT);
        let h2 = page_content_hash(&[5, 6, 7, 8], PageHash::ROOT);

        cache.insert(h1, vec![PageId(0)], 4, page_size);
        cache.insert(h2, vec![PageId(1)], 4, page_size);

        // Touch h1 by looking it up.
        cache.lookup(&[1, 2, 3, 4], page_size);

        // Now h2 should be LRU.
        let evicted = cache.evict_lru().unwrap();
        assert_eq!(evicted, vec![PageId(1)]);
    }

    #[test]
    fn evict_empty_cache_returns_none() {
        let mut cache = PrefixCache::new(true);
        assert!(cache.evict_lru().is_none());
    }

    #[test]
    fn duplicate_insert_only_touches_lru() {
        let mut cache = PrefixCache::new(true);
        let page_size = 4;

        let h1 = page_content_hash(&[1, 2, 3, 4], PageHash::ROOT);
        cache.insert(h1, vec![PageId(0)], 4, page_size);
        cache.insert(h1, vec![PageId(99)], 4, page_size); // duplicate

        assert_eq!(cache.len(), 1);

        // Should still return the original page IDs.
        let result = cache.lookup(&[1, 2, 3, 4], page_size);
        assert_eq!(result.matched_pages[0], vec![PageId(0)]);
    }
}
