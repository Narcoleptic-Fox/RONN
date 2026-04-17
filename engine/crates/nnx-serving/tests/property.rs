use nnx_serving::block_manager::BlockAllocator;
use proptest::prelude::*;

proptest! {
    #[test]
    fn allocator_allocate_free_preserves_page_counts(total_pages in 1usize..32, requested_pages in 0usize..32) {
        let alloc_count = requested_pages.min(total_pages);
        let mut allocator = BlockAllocator::new(total_pages, 4, 2, 4);
        let mut pages = Vec::with_capacity(alloc_count);

        for _ in 0..alloc_count {
            pages.push(allocator.allocate().unwrap());
        }

        let stats = allocator.stats();
        prop_assert_eq!(stats.total_pages, total_pages);
        prop_assert_eq!(stats.used_pages, alloc_count);
        prop_assert_eq!(stats.free_pages + stats.used_pages, stats.total_pages);

        for page in pages {
            allocator.dec_ref(page).unwrap();
        }

        let final_stats = allocator.stats();
        prop_assert_eq!(final_stats.used_pages, 0);
        prop_assert_eq!(final_stats.free_pages, total_pages);
    }
}