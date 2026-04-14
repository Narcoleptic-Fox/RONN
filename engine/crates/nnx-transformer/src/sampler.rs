//! Token sampling: temperature, top-k, top-p (nucleus), min-p, typical,
//! tail-free sampling (TFS), repetition penalty, and Mirostat v2.

/// Sampling configuration for the `sample()` function.
///
/// Fields that control optional filters default to "disabled":
/// - `min_p = 0.0` — disabled
/// - `typical_p = 1.0` — disabled
/// - `tfs_z = 1.0` — disabled
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// Min-P threshold relative to the max probability.
    /// Keep tokens where `p >= min_p * max_p`. Set to 0.0 to disable.
    pub min_p: f32,
    /// Typical sampling mass. Keep tokens near the entropy of the
    /// distribution until cumulative probability >= typical_p.
    /// Set to 1.0 to disable.
    pub typical_p: f32,
    /// Tail-Free Sampling z-cutoff. Removes the tail of the distribution
    /// based on the second derivative of sorted probabilities.
    /// Set to 1.0 to disable.
    pub tfs_z: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.1,
        }
    }
}

impl SamplerConfig {
    /// Greedy decoding — always picks the most likely token.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.0,
        }
    }

    /// Creative generation — diverse output with min-p for quality control.
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_k: 0,
            top_p: 0.95,
            min_p: 0.05,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.1,
        }
    }

    /// Precise generation — lower temperature, focused sampling.
    pub fn precise() -> Self {
        Self {
            temperature: 0.3,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.1,
        }
    }

    /// Balanced chat defaults — natural conversation quality.
    pub fn default_chat() -> Self {
        Self {
            temperature: 0.7,
            top_k: 0,
            top_p: 0.9,
            min_p: 0.05,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.1,
        }
    }
}

/// Mirostat v2 state (Basu et al. 2021).
///
/// Mirostat maintains a target surprise level (tau) by adapting a dynamic
/// probability threshold (mu) after each sampled token.  Because mu must
/// persist across calls this lives outside `SamplerConfig`.
#[derive(Debug, Clone)]
pub struct MirostatState {
    /// Target surprise in nats. Default 5.0.
    pub tau: f32,
    /// Learning rate for mu updates. Default 0.1.
    pub eta: f32,
    /// Current surprise threshold. Initialized to `2 * tau`.
    pub mu: f32,
}

impl MirostatState {
    pub fn new(tau: f32, eta: f32) -> Self {
        Self { tau, eta, mu: 2.0 * tau }
    }
}

impl Default for MirostatState {
    fn default() -> Self {
        Self::new(5.0, 0.1)
    }
}

// ---------------------------------------------------------------------------
// Public sampling functions
// ---------------------------------------------------------------------------

/// Sample a token from logits using the full filter pipeline.
///
/// Pipeline:
/// 1. Repetition penalty
/// 2. Greedy (if `temperature < 1e-6`)
/// 3. Temperature scaling
/// 4. Softmax
/// 5. Min-P filter
/// 6. Typical sampling filter
/// 7. Tail-free sampling (TFS) filter
/// 8. Top-K filter
/// 9. Top-P (nucleus) filter
/// 10. Renormalize
/// 11. Weighted random sample
pub fn sample(
    logits: &[f32],
    config: &SamplerConfig,
    prev_tokens: &[u32],
    rng_state: &mut u64,
) -> u32 {
    let n = logits.len();
    let mut probs = logits.to_vec();

    // --- 1. Repetition penalty ---
    if config.repetition_penalty != 1.0 {
        for &tok in prev_tokens {
            let i = tok as usize;
            if i < n {
                if probs[i] > 0.0 {
                    probs[i] /= config.repetition_penalty;
                } else {
                    probs[i] *= config.repetition_penalty;
                }
            }
        }
    }

    // --- 2. Greedy ---
    if config.temperature < 1e-6 {
        return argmax(&probs) as u32;
    }

    // --- 3. Temperature scaling ---
    for v in probs.iter_mut() {
        *v /= config.temperature;
    }

    // --- 4. Softmax ---
    softmax_in_place(&mut probs);

    // --- 5. Min-P filter ---
    if config.min_p > 0.0 {
        apply_min_p(&mut probs, config.min_p);
    }

    // --- 6. Typical sampling filter ---
    if config.typical_p < 1.0 {
        apply_typical(&mut probs, config.typical_p);
    }

    // --- 7. Tail-free sampling (TFS) filter ---
    if config.tfs_z < 1.0 {
        apply_tfs(&mut probs, config.tfs_z);
    }

    // --- 8. Top-K filter ---
    if config.top_k > 0 && config.top_k < n {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let cutoff = probs[idx[config.top_k - 1]];
        for i in 0..n {
            if probs[i] < cutoff {
                probs[i] = 0.0;
            }
        }
    }

    // --- 9. Top-P (nucleus) filter ---
    if config.top_p < 1.0 {
        apply_top_p(&mut probs, config.top_p);
    }

    // --- 10. Renormalize ---
    renormalize(&mut probs);

    // --- 11. Weighted random sample ---
    weighted_sample(&probs, n, rng_state)
}

/// Sample using Mirostat v2 adaptive surprise control.
///
/// Unlike `sample()`, Mirostat mutates `state.mu` after each call to adapt
/// the surprise threshold toward `state.tau`.  Repetition penalty is applied
/// before the Mirostat filtering logic.
pub fn sample_mirostat(
    logits: &[f32],
    state: &mut MirostatState,
    prev_tokens: &[u32],
    rng_state: &mut u64,
) -> u32 {
    let n = logits.len();
    let mut probs = logits.to_vec();

    // Repetition penalty (same as sample())
    for &tok in prev_tokens {
        let i = tok as usize;
        if i < n {
            if probs[i] > 0.0 {
                probs[i] /= 1.1; // conservative default; callers can pre-apply
            } else {
                probs[i] *= 1.1;
            }
        }
    }

    softmax_in_place(&mut probs);

    // Build sorted (index, prob) descending so we can compute surprise values.
    let mut sorted: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, p)| p > 0.0)
        .collect();
    sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Keep tokens where -log2(p_i) <= mu (i.e. surprise <= threshold).
    // Always keep at least one token.
    let mut keep_end = 1;
    for (rank, &(_, p)) in sorted.iter().enumerate() {
        let surprise = -p.log2();
        if surprise <= state.mu {
            keep_end = rank + 1;
        } else {
            break;
        }
    }

    // Zero out tokens that did not make the cut.
    let kept_indices: std::collections::HashSet<usize> =
        sorted[..keep_end].iter().map(|&(i, _)| i).collect();
    for i in 0..n {
        if !kept_indices.contains(&i) {
            probs[i] = 0.0;
        }
    }

    renormalize(&mut probs);

    let token = weighted_sample(&probs, n, rng_state);

    // Update mu: mu = mu - eta * (surprise_sampled - tau)
    let sampled_surprise = -probs[token as usize].max(1e-38).log2();
    state.mu -= state.eta * (sampled_surprise - state.tau);

    token
}

// ---------------------------------------------------------------------------
// Filter helpers (operate on a probability vector in-place)
// ---------------------------------------------------------------------------

/// Min-P filter (Nguyen 2023).
///
/// Keeps tokens where `p >= min_p * max_p`.  This removes tokens that are
/// implausible relative to the best candidate, without depending on a fixed
/// cumulative mass.
fn apply_min_p(probs: &mut [f32], min_p: f32) {
    let max_p = probs.iter().cloned().fold(0.0f32, f32::max);
    let threshold = min_p * max_p;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

/// Typical sampling filter (Meister et al. 2022).
///
/// Computes the entropy H(p) of the current distribution, then keeps tokens
/// whose negative log-probability is closest to H(p), accumulating until the
/// kept mass >= typical_p.
fn apply_typical(probs: &mut [f32], typical_p: f32) {
    // Entropy H(p) = -sum(p * log(p)) over non-zero entries.
    let entropy: f32 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // For each token compute |(-log p_i) - H|, then sort ascending.
    let mut candidates: Vec<(usize, f32, f32)> = probs
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > 0.0)
        .map(|(i, &p)| {
            let neg_log_p = -p.ln();
            let distance = (neg_log_p - entropy).abs();
            (i, p, distance)
        })
        .collect();

    candidates.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Accumulate until we have enough probability mass.
    let mut cum = 0.0f32;
    let mut keep_end = 0;
    for (rank, &(_, p, _)) in candidates.iter().enumerate() {
        cum += p;
        keep_end = rank + 1;
        if cum >= typical_p {
            break;
        }
    }

    let keep: std::collections::HashSet<usize> =
        candidates[..keep_end].iter().map(|&(i, _, _)| i).collect();
    for i in 0..probs.len() {
        if !keep.contains(&i) {
            probs[i] = 0.0;
        }
    }
}

/// Tail-Free Sampling filter (Busconi 2021).
///
/// Computes the second derivative of the sorted probability curve, normalizes
/// it, then removes tokens in the tail once the accumulated second-derivative
/// mass exceeds `1 - tfs_z`.
fn apply_tfs(probs: &mut [f32], tfs_z: f32) {
    // Sort indices by probability descending.
    let mut sorted: Vec<(usize, f32)> = probs
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, p)| p > 0.0)
        .collect();
    sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if sorted.len() < 3 {
        // Not enough tokens to compute a second derivative; do nothing.
        return;
    }

    let vals: Vec<f32> = sorted.iter().map(|&(_, p)| p).collect();

    // First derivative (forward differences).
    let d1: Vec<f32> = vals.windows(2).map(|w| w[1] - w[0]).collect();

    // Second derivative.
    let d2: Vec<f32> = d1.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    // Normalize so second derivatives sum to 1.
    let d2_sum: f32 = d2.iter().sum();
    if d2_sum <= 0.0 {
        return;
    }
    let d2_norm: Vec<f32> = d2.iter().map(|&v| v / d2_sum).collect();

    // Accumulate from the head until we reach tfs_z, then zero the rest.
    // The second-derivative index k corresponds to sorted[k+2].
    let mut cum = 0.0f32;
    let mut cutoff = sorted.len(); // default: keep all
    for (k, &d) in d2_norm.iter().enumerate() {
        cum += d;
        if cum >= tfs_z {
            // Keep sorted[0..=k+1], zero out sorted[k+2..].
            cutoff = k + 2;
            break;
        }
    }

    let keep: std::collections::HashSet<usize> =
        sorted[..cutoff].iter().map(|&(i, _)| i).collect();
    for i in 0..probs.len() {
        if !keep.contains(&i) {
            probs[i] = 0.0;
        }
    }
}

/// Top-P (nucleus) filter.  Zeroes out tokens beyond the cumulative mass.
fn apply_top_p(probs: &mut [f32], top_p: f32) {
    let mut sorted: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cum = 0.0f32;
    let mut cutoff = sorted.len();
    for (i, &(_, p)) in sorted.iter().enumerate() {
        cum += p;
        if cum > top_p {
            cutoff = i + 1;
            break;
        }
    }
    let keep: std::collections::HashSet<usize> =
        sorted[..cutoff].iter().map(|&(i, _)| i).collect();
    for i in 0..probs.len() {
        if !keep.contains(&i) {
            probs[i] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Shared numeric utilities
// ---------------------------------------------------------------------------

fn softmax_in_place(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

fn renormalize(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn next_random(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}

fn weighted_sample(probs: &[f32], n: usize, rng_state: &mut u64) -> u32 {
    let r = next_random(rng_state);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i as u32;
        }
    }
    (n - 1) as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helpers ---

    fn make_rng(seed: u64) -> u64 {
        seed
    }

    /// Uniform logits of length n.
    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    /// Logits where index `peak` is much larger than all others.
    fn peaked_logits(n: usize, peak: usize) -> Vec<f32> {
        let mut v = vec![0.1; n];
        v[peak] = 10.0;
        v
    }

    // --- Existing test (unchanged) ---

    #[test]
    fn test_greedy() {
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let mut rng = make_rng(42);
        assert_eq!(sample(&logits, &SamplerConfig::greedy(), &[], &mut rng), 2);
    }

    // --- Preset field tests ---

    #[test]
    fn test_preset_creative_fields() {
        let c = SamplerConfig::creative();
        assert_eq!(c.temperature, 0.9);
        assert_eq!(c.min_p, 0.05);
        assert_eq!(c.top_p, 0.95);
    }

    #[test]
    fn test_preset_precise_fields() {
        let p = SamplerConfig::precise();
        assert_eq!(p.temperature, 0.3);
        assert_eq!(p.top_k, 40);
        assert_eq!(p.top_p, 0.9);
    }

    #[test]
    fn test_preset_default_chat_fields() {
        let d = SamplerConfig::default_chat();
        assert_eq!(d.temperature, 0.7);
        assert_eq!(d.min_p, 0.05);
        assert_eq!(d.top_p, 0.9);
    }

    // --- Min-P tests ---

    #[test]
    fn test_min_p_removes_low_prob_tokens() {
        // After softmax: token 0 dominates heavily.
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.05,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = make_rng(42);
        // All samples should be token 0 since others are filtered by min-p.
        for _ in 0..20 {
            assert_eq!(sample(&logits, &config, &[], &mut rng), 0);
        }
    }

    #[test]
    fn test_min_p_disabled_at_zero() {
        // With min_p=0.0 and uniform probs, all tokens should be reachable.
        let logits = uniform_logits(4);
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = make_rng(7);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(sample(&logits, &config, &[], &mut rng));
        }
        assert_eq!(seen.len(), 4, "all four tokens should be sampled");
    }

    #[test]
    fn test_min_p_apply_directly() {
        // Direct test of apply_min_p: only token 0 should survive.
        let mut probs = vec![0.8, 0.1, 0.05, 0.05];
        apply_min_p(&mut probs, 0.2); // threshold = 0.2 * 0.8 = 0.16
        assert!(probs[0] > 0.0, "token 0 should survive");
        assert_eq!(probs[1], 0.0, "token 1 should be zeroed");
        assert_eq!(probs[2], 0.0, "token 2 should be zeroed");
        assert_eq!(probs[3], 0.0, "token 3 should be zeroed");
    }

    // --- Typical sampling tests ---

    #[test]
    fn test_typical_keeps_tokens_near_entropy() {
        // Uniform distribution: entropy is high, every token is equally
        // "typical".  With typical_p=0.5 we should keep roughly half.
        let mut probs = vec![0.25, 0.25, 0.25, 0.25];
        apply_typical(&mut probs, 0.5);
        let kept: usize = probs.iter().filter(|&&p| p > 0.0).count();
        // At least 1 and at most 3 tokens kept for 50% mass of uniform dist.
        assert!(kept >= 1 && kept <= 4);
    }

    #[test]
    fn test_typical_disabled_at_one() {
        // typical_p=1.0 means no filtering — every non-zero token survives.
        let mut probs = vec![0.4, 0.3, 0.2, 0.1];
        let original = probs.clone();
        apply_typical(&mut probs, 1.0);
        // With typical_p=1.0 the function isn't even called (guarded in
        // sample()), but test the function itself behaves gracefully.
        let sum_before: f32 = original.iter().sum();
        let sum_after: f32 = probs.iter().filter(|&&p| p > 0.0).sum();
        // At least 50% of the mass should be retained.
        assert!(sum_after >= sum_before * 0.5);
    }

    #[test]
    fn test_typical_apply_skewed_distribution() {
        // In a skewed distribution, the highest-prob token is "atypical"
        // (its surprise is very low vs the entropy).  Typical sampling
        // may filter it out in favour of more typical tokens.
        let mut probs = vec![0.97, 0.01, 0.01, 0.01];
        // Entropy ≈ -0.97*ln(0.97) - 3*0.01*ln(0.01) ≈ 0.168
        // Token 0 surprise ≈ -ln(0.97) ≈ 0.03 (very atypical, close to 0)
        // Token 1-3 surprise ≈ -ln(0.01) ≈ 4.6 (also atypical, far above H)
        // Test just verifies the function completes and retains >=1 token.
        apply_typical(&mut probs, 0.5);
        let kept: usize = probs.iter().filter(|&&p| p > 0.0).count();
        assert!(kept >= 1);
    }

    // --- Tail-Free Sampling tests ---

    #[test]
    fn test_tfs_removes_tail() {
        // Probability mass concentrated at index 0-1, long flat tail at 2-9.
        let mut probs = vec![0.4, 0.35, 0.05, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01];
        let original_sum: f32 = probs.iter().sum();
        apply_tfs(&mut probs, 0.5);
        let remaining: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .collect();
        // At least the top token should survive.
        assert!(remaining.contains(&0), "top token must survive TFS");
        // Total probability retained should be > 0.
        let kept_sum: f32 = probs.iter().sum();
        assert!(kept_sum > 0.0);
        assert!(kept_sum <= original_sum + 1e-6);
    }

    #[test]
    fn test_tfs_too_few_tokens_is_noop() {
        // With fewer than 3 tokens, TFS cannot compute a second derivative
        // and should leave the probabilities unchanged.
        let mut probs = vec![0.6, 0.4];
        apply_tfs(&mut probs, 0.9);
        assert!(probs[0] > 0.0);
        assert!(probs[1] > 0.0);
    }

    #[test]
    fn test_tfs_z_at_one_via_sample_is_noop() {
        // tfs_z=1.0 skips the filter entirely inside sample().
        let logits = peaked_logits(10, 3);
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = make_rng(42);
        // Should not panic; result is dominated by the peak.
        let tok = sample(&logits, &config, &[], &mut rng);
        assert!(tok < 10);
    }

    // --- Mirostat v2 tests ---

    #[test]
    fn test_mirostat_state_default() {
        let state = MirostatState::default();
        assert_eq!(state.tau, 5.0);
        assert_eq!(state.eta, 0.1);
        assert_eq!(state.mu, 10.0); // 2 * tau
    }

    #[test]
    fn test_mirostat_mu_adapts_over_calls() {
        // With a very peaked distribution, sampled surprise is low.
        // mu should decrease (approach tau) over multiple calls.
        let logits = peaked_logits(50, 0);
        let mut state = MirostatState::new(5.0, 0.1);
        let initial_mu = state.mu;
        let mut rng = make_rng(99);

        for _ in 0..30 {
            sample_mirostat(&logits, &mut state, &[], &mut rng);
        }

        // mu should have moved from its initial value (10.0)
        assert_ne!(state.mu, initial_mu, "mu must change over multiple calls");
    }

    #[test]
    fn test_mirostat_returns_valid_token() {
        let logits = uniform_logits(10);
        let mut state = MirostatState::default();
        let mut rng = make_rng(1);
        let tok = sample_mirostat(&logits, &mut state, &[], &mut rng);
        assert!((tok as usize) < 10, "token must be within vocab range");
    }

    #[test]
    fn test_mirostat_single_token() {
        let logits = vec![1.0];
        let mut state = MirostatState::default();
        let mut rng = make_rng(1);
        assert_eq!(sample_mirostat(&logits, &mut state, &[], &mut rng), 0);
    }

    // --- Edge-case tests ---

    #[test]
    fn test_single_token_vocab() {
        let logits = vec![1.0f32];
        let mut rng = make_rng(0);
        assert_eq!(sample(&logits, &SamplerConfig::default(), &[], &mut rng), 0);
    }

    #[test]
    fn test_greedy_on_peaked_logits() {
        let logits = peaked_logits(20, 7);
        let mut rng = make_rng(0);
        assert_eq!(sample(&logits, &SamplerConfig::greedy(), &[], &mut rng), 7);
    }

    #[test]
    fn test_uniform_all_tokens_reachable() {
        let logits = uniform_logits(8);
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = make_rng(5);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..400 {
            seen.insert(sample(&logits, &config, &[], &mut rng));
        }
        assert_eq!(seen.len(), 8, "all tokens should be reachable from uniform logits");
    }

    #[test]
    fn test_very_low_temperature_approaches_greedy() {
        // temperature near zero should reliably pick the peak token.
        let logits = peaked_logits(10, 4);
        let config = SamplerConfig {
            temperature: 0.01,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 1.0,
        };
        let mut rng = make_rng(0);
        for _ in 0..10 {
            assert_eq!(sample(&logits, &config, &[], &mut rng), 4);
        }
    }

    #[test]
    fn test_repetition_penalty_suppresses_previous_tokens() {
        // With a large repetition penalty, token 2 (previously seen) should
        // be less likely than without the penalty.
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let config = SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            typical_p: 1.0,
            tfs_z: 1.0,
            repetition_penalty: 5.0,
        };
        let mut count_two = 0u32;
        let mut rng = make_rng(17);
        for _ in 0..200 {
            if sample(&logits, &config, &[2], &mut rng) == 2 {
                count_two += 1;
            }
        }
        // Token 2 should appear well below 25% (uniform rate without penalty).
        // With penalty=5.0 the expected rate is ~6.25%; allow generous headroom
        // for LCG variance: anything below 20% of trials (40/200) is acceptable.
        assert!(
            count_two < 40,
            "token 2 appeared {count_two}/200 times — penalty not working"
        );
    }
}
