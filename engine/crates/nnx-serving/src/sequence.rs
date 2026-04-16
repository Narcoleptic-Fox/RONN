//! Sequence state machine for continuous batching.
//!
//! Each inference request is tracked as a [`Sequence`] that moves through
//! a state machine: Waiting → Prefilling → Decoding → Finished.

use crate::paged_cache::SequencePageTable;

/// Unique identifier for a sequence in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u64);

/// Why a sequence finished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit the end-of-sequence token.
    EndOfSequence,
    /// Reached the maximum generation length.
    MaxTokens,
    /// Cancelled by the user.
    Cancelled,
}

/// Current state of a sequence in the scheduler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceState {
    /// Waiting in the queue, not yet admitted.
    Waiting,
    /// Actively prefilling prompt tokens.
    Prefilling {
        /// Number of prompt tokens remaining to prefill.
        tokens_remaining: usize,
    },
    /// Actively generating tokens (decode phase).
    Decoding,
    /// Generation complete.
    Finished { reason: FinishReason },
    /// Preempted: pages have been reclaimed, sequence must be re-prefilled.
    Preempted,
}

/// A sequence being tracked by the scheduler.
#[derive(Debug)]
pub struct Sequence {
    /// Unique identifier.
    pub id: SequenceId,
    /// Current state.
    pub state: SequenceState,
    /// Original prompt token IDs.
    pub prompt_tokens: Vec<u32>,
    /// Generated token IDs (appended during decode).
    pub generated_tokens: Vec<u32>,
    /// Per-layer page table for this sequence's KV cache.
    pub page_table: SequencePageTable,
    /// Maximum number of tokens to generate (excluding prompt).
    pub max_new_tokens: usize,
    /// Number of prompt tokens that have been prefilled so far.
    pub prefilled_tokens: usize,
    /// Priority (lower = higher priority). Default 0.
    pub priority: u32,
}

impl Sequence {
    /// Create a new sequence in the Waiting state.
    pub fn new(
        id: SequenceId,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        num_layers: usize,
        page_size: usize,
    ) -> Self {
        Self {
            id,
            state: SequenceState::Waiting,
            prompt_tokens,
            generated_tokens: Vec::new(),
            page_table: SequencePageTable::new(num_layers, page_size),
            max_new_tokens,
            prefilled_tokens: 0,
            priority: 0,
        }
    }

    /// Total number of tokens this sequence will occupy at maximum.
    pub fn max_total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.max_new_tokens
    }

    /// Number of tokens currently in the KV cache.
    pub fn cached_tokens(&self) -> usize {
        self.page_table.num_tokens()
    }

    /// Total tokens generated so far.
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    /// Whether this sequence has finished generation.
    pub fn is_finished(&self) -> bool {
        matches!(self.state, SequenceState::Finished { .. })
    }

    /// Whether this sequence is actively running (prefilling or decoding).
    pub fn is_running(&self) -> bool {
        matches!(
            self.state,
            SequenceState::Prefilling { .. } | SequenceState::Decoding
        )
    }

    /// Transition to prefilling state (or straight to decoding if fully cached).
    pub fn start_prefill(&mut self) {
        let remaining = self
            .prompt_tokens
            .len()
            .saturating_sub(self.prefilled_tokens);
        if remaining == 0 {
            // Entire prompt was satisfied by prefix cache — skip prefill.
            self.state = SequenceState::Decoding;
        } else {
            self.state = SequenceState::Prefilling {
                tokens_remaining: remaining,
            };
        }
    }

    /// Record that `n` tokens were prefilled in this iteration.
    pub fn advance_prefill(&mut self, n: usize) {
        self.prefilled_tokens += n;
        let remaining = self
            .prompt_tokens
            .len()
            .saturating_sub(self.prefilled_tokens);
        if remaining == 0 {
            self.state = SequenceState::Decoding;
        } else {
            self.state = SequenceState::Prefilling {
                tokens_remaining: remaining,
            };
        }
    }

    /// Record a generated token.
    pub fn append_token(&mut self, token_id: u32) {
        self.generated_tokens.push(token_id);
    }

    /// Mark as finished.
    pub fn finish(&mut self, reason: FinishReason) {
        self.state = SequenceState::Finished { reason };
    }

    /// Check if generation should stop (max tokens reached).
    pub fn should_stop(&self) -> bool {
        self.generated_tokens.len() >= self.max_new_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sequence_is_waiting() {
        let seq = Sequence::new(SequenceId(1), vec![1, 2, 3], 10, 4, 16);
        assert_eq!(seq.state, SequenceState::Waiting);
        assert_eq!(seq.prompt_tokens.len(), 3);
        assert!(!seq.is_finished());
        assert!(!seq.is_running());
    }

    #[test]
    fn prefill_transitions() {
        let mut seq = Sequence::new(SequenceId(1), vec![1, 2, 3, 4, 5], 10, 4, 16);
        seq.start_prefill();
        assert_eq!(
            seq.state,
            SequenceState::Prefilling {
                tokens_remaining: 5
            }
        );
        assert!(seq.is_running());

        seq.advance_prefill(3);
        assert_eq!(
            seq.state,
            SequenceState::Prefilling {
                tokens_remaining: 2
            }
        );

        seq.advance_prefill(2);
        assert_eq!(seq.state, SequenceState::Decoding);
    }

    #[test]
    fn decode_and_finish() {
        let mut seq = Sequence::new(SequenceId(1), vec![1], 3, 4, 16);
        seq.start_prefill();
        seq.advance_prefill(1);
        assert_eq!(seq.state, SequenceState::Decoding);

        seq.append_token(10);
        seq.append_token(11);
        assert!(!seq.should_stop());

        seq.append_token(12);
        assert!(seq.should_stop());

        seq.finish(FinishReason::MaxTokens);
        assert!(seq.is_finished());
        assert_eq!(seq.generated_tokens, vec![10, 11, 12]);
    }

    #[test]
    fn max_total_tokens() {
        let seq = Sequence::new(SequenceId(1), vec![1, 2, 3], 20, 4, 16);
        assert_eq!(seq.max_total_tokens(), 23);
    }
}
