//! Token sampling: temperature, top-k, top-p (nucleus), repetition penalty.

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
        }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Sample a token from logits.
pub fn sample(
    logits: &[f32],
    config: &SamplerConfig,
    prev_tokens: &[u32],
    rng_state: &mut u64,
) -> u32 {
    let n = logits.len();
    let mut probs = logits.to_vec();

    // Repetition penalty
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

    // Greedy
    if config.temperature < 1e-6 {
        return argmax(&probs) as u32;
    }

    // Temperature
    for v in probs.iter_mut() {
        *v /= config.temperature;
    }

    // Softmax
    let max = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in probs.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in probs.iter_mut() {
        *v /= sum;
    }

    // Top-k
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

    // Top-p
    if config.top_p < 1.0 {
        let mut sorted: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0f32;
        let mut cutoff = sorted.len();
        for (i, &(_, p)) in sorted.iter().enumerate() {
            cum += p;
            if cum > config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        let keep: std::collections::HashSet<usize> =
            sorted[..cutoff].iter().map(|&(i, _)| i).collect();
        for i in 0..n {
            if !keep.contains(&i) {
                probs[i] = 0.0;
            }
        }
    }

    // Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }

    // Sample
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy() {
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let mut rng = 42u64;
        assert_eq!(sample(&logits, &SamplerConfig::greedy(), &[], &mut rng), 2);
    }
}
