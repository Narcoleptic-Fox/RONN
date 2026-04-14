//! State Space Model (SSM) operations for Mamba architecture.
//!
//! Mamba uses a selective scan mechanism instead of attention.
//! Core recurrence: x[t] = A * x[t-1] + B * u[t], y[t] = C * x[t] + D * u[t]

use nnx_core::error::EngineError;

/// Selective scan -- the core Mamba operation.
///
/// Runs a discretized state space model:
///   x[t] = A_bar[t] * x[t-1] + B_bar[t] * u[t]
///   y[t] = C[t] * x[t] + D * u[t]
///
/// where A_bar and B_bar are the discretized (input-dependent) parameters.
///
/// `u`:      input sequence [seq_len, d_inner]
/// `a_bar`:  discretized A [seq_len, d_inner, d_state]
/// `b_bar`:  discretized B [seq_len, d_inner, d_state]
/// `c`:      output projection [seq_len, d_inner, d_state]
/// `d`:      skip connection [d_inner]
/// `output`: [seq_len, d_inner]
///
/// d_state is the SSM state dimension (typically 16).
pub fn selective_scan_f32(
    u: &[f32],
    a_bar: &[f32],
    b_bar: &[f32],
    c: &[f32],
    d: &[f32],
    output: &mut [f32],
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
) {
    assert_eq!(u.len(), seq_len * d_inner);
    assert_eq!(a_bar.len(), seq_len * d_inner * d_state);
    assert_eq!(b_bar.len(), seq_len * d_inner * d_state);
    assert_eq!(c.len(), seq_len * d_inner * d_state);
    assert_eq!(d.len(), d_inner);
    assert_eq!(output.len(), seq_len * d_inner);

    // State: x[d_inner, d_state] -- initialized to zero
    let mut state = vec![0.0f32; d_inner * d_state];

    for t in 0..seq_len {
        for di in 0..d_inner {
            let u_val = u[t * d_inner + di];
            let mut y_val = 0.0f32;

            for ds in 0..d_state {
                let idx = t * d_inner * d_state + di * d_state + ds;
                let state_idx = di * d_state + ds;

                // x[t] = A_bar[t] * x[t-1] + B_bar[t] * u[t]
                state[state_idx] = a_bar[idx] * state[state_idx] + b_bar[idx] * u_val;

                // y[t] += C[t] * x[t]
                y_val += c[idx] * state[state_idx];
            }

            // y[t] += D * u[t]  (skip connection)
            output[t * d_inner + di] = y_val + d[di] * u_val;
        }
    }
}

/// Checked version of `selective_scan_f32`.
pub fn selective_scan_f32_checked(
    u: &[f32],
    a_bar: &[f32],
    b_bar: &[f32],
    c: &[f32],
    d: &[f32],
    output: &mut [f32],
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
) -> nnx_core::error::Result<()> {
    let expected_u = seq_len * d_inner;
    let expected_abc = seq_len * d_inner * d_state;
    if u.len() != expected_u {
        return Err(EngineError::ShapeMismatch(format!(
            "selective_scan: u.len()={} but seq_len*d_inner={}",
            u.len(),
            expected_u
        )));
    }
    if a_bar.len() != expected_abc {
        return Err(EngineError::ShapeMismatch(format!(
            "selective_scan: a_bar.len()={} but expected {}",
            a_bar.len(),
            expected_abc
        )));
    }
    if b_bar.len() != expected_abc {
        return Err(EngineError::ShapeMismatch(format!(
            "selective_scan: b_bar.len()={} but expected {}",
            b_bar.len(),
            expected_abc
        )));
    }
    if c.len() != expected_abc {
        return Err(EngineError::ShapeMismatch(format!(
            "selective_scan: c.len()={} but expected {}",
            c.len(),
            expected_abc
        )));
    }
    if d.len() != d_inner {
        return Err(EngineError::ShapeMismatch(format!(
            "selective_scan: d.len()={} but d_inner={}",
            d.len(),
            d_inner
        )));
    }
    if output.len() != expected_u {
        return Err(EngineError::ShapeMismatch(format!(
            "selective_scan: output.len()={} but seq_len*d_inner={}",
            output.len(),
            expected_u
        )));
    }
    selective_scan_f32(u, a_bar, b_bar, c, d, output, seq_len, d_inner, d_state);
    Ok(())
}

/// Causal Conv1D -- 1D convolution that only looks at past values.
///
/// This is the depthwise causal convolution used in Mamba before the SSM.
/// Each channel is convolved independently (depthwise).
///
/// `input`: [seq_len, d_inner] -- input sequence
/// `weight`: [d_inner, kernel_size] -- per-channel conv weights
/// `bias`: [d_inner] -- optional per-channel bias
/// `output`: [seq_len, d_inner]
///
/// The convolution is left-padded so output[t] only depends on
/// input[t-kernel_size+1..=t].
pub fn causal_conv1d_f32(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    seq_len: usize,
    d_inner: usize,
    kernel_size: usize,
) {
    assert_eq!(input.len(), seq_len * d_inner);
    assert_eq!(weight.len(), d_inner * kernel_size);
    assert_eq!(output.len(), seq_len * d_inner);

    for t in 0..seq_len {
        for di in 0..d_inner {
            let mut sum = bias.map_or(0.0, |b| b[di]);

            for k in 0..kernel_size {
                // Causal: look back k positions
                let src_t = t as isize - k as isize;
                if src_t >= 0 {
                    let src_val = input[src_t as usize * d_inner + di];
                    sum += src_val * weight[di * kernel_size + k];
                }
                // Implicit zero-padding for negative indices
            }

            output[t * d_inner + di] = sum;
        }
    }
}

/// Checked version of `causal_conv1d_f32`.
pub fn causal_conv1d_f32_checked(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    seq_len: usize,
    d_inner: usize,
    kernel_size: usize,
) -> nnx_core::error::Result<()> {
    if input.len() != seq_len * d_inner {
        return Err(EngineError::ShapeMismatch(format!(
            "causal_conv1d: input.len()={} but seq_len*d_inner={}",
            input.len(),
            seq_len * d_inner
        )));
    }
    if weight.len() != d_inner * kernel_size {
        return Err(EngineError::ShapeMismatch(format!(
            "causal_conv1d: weight.len()={} but d_inner*kernel_size={}",
            weight.len(),
            d_inner * kernel_size
        )));
    }
    if output.len() != seq_len * d_inner {
        return Err(EngineError::ShapeMismatch(format!(
            "causal_conv1d: output.len()={} but seq_len*d_inner={}",
            output.len(),
            seq_len * d_inner
        )));
    }
    if let Some(b) = bias {
        if b.len() != d_inner {
            return Err(EngineError::ShapeMismatch(format!(
                "causal_conv1d: bias.len()={} but d_inner={}",
                b.len(),
                d_inner
            )));
        }
    }
    causal_conv1d_f32(input, weight, bias, output, seq_len, d_inner, kernel_size);
    Ok(())
}

/// Discretize continuous SSM parameters for selective scan.
///
/// Given continuous A [d_inner, d_state] and delta [seq_len, d_inner]:
///   A_bar[t] = exp(delta[t] * A)
///   B_bar[t] = delta[t] * B[t]  (simplified Euler discretization)
///
/// `a`: continuous A parameter [d_inner, d_state]
/// `b`: input-dependent B [seq_len, d_inner, d_state]
/// `delta`: discretization step [seq_len, d_inner]
/// `a_bar`: output [seq_len, d_inner, d_state]
/// `b_bar`: output [seq_len, d_inner, d_state]
pub fn discretize_ssm_f32(
    a: &[f32],
    b: &[f32],
    delta: &[f32],
    a_bar: &mut [f32],
    b_bar: &mut [f32],
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
) {
    assert_eq!(a.len(), d_inner * d_state);
    assert_eq!(b.len(), seq_len * d_inner * d_state);
    assert_eq!(delta.len(), seq_len * d_inner);
    assert_eq!(a_bar.len(), seq_len * d_inner * d_state);
    assert_eq!(b_bar.len(), seq_len * d_inner * d_state);

    for t in 0..seq_len {
        for di in 0..d_inner {
            let dt = delta[t * d_inner + di];
            for ds in 0..d_state {
                let a_idx = di * d_state + ds;
                let idx = t * d_inner * d_state + di * d_state + ds;
                a_bar[idx] = (dt * a[a_idx]).exp();
                b_bar[idx] = dt * b[idx];
            }
        }
    }
}

/// Checked version of `discretize_ssm_f32`.
pub fn discretize_ssm_f32_checked(
    a: &[f32],
    b: &[f32],
    delta: &[f32],
    a_bar: &mut [f32],
    b_bar: &mut [f32],
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
) -> nnx_core::error::Result<()> {
    let expected_a = d_inner * d_state;
    let expected_b = seq_len * d_inner * d_state;
    let expected_delta = seq_len * d_inner;
    if a.len() != expected_a {
        return Err(EngineError::ShapeMismatch(format!(
            "discretize_ssm: a.len()={} but d_inner*d_state={}",
            a.len(),
            expected_a
        )));
    }
    if b.len() != expected_b {
        return Err(EngineError::ShapeMismatch(format!(
            "discretize_ssm: b.len()={} but expected {}",
            b.len(),
            expected_b
        )));
    }
    if delta.len() != expected_delta {
        return Err(EngineError::ShapeMismatch(format!(
            "discretize_ssm: delta.len()={} but seq_len*d_inner={}",
            delta.len(),
            expected_delta
        )));
    }
    if a_bar.len() != expected_b {
        return Err(EngineError::ShapeMismatch(format!(
            "discretize_ssm: a_bar.len()={} but expected {}",
            a_bar.len(),
            expected_b
        )));
    }
    if b_bar.len() != expected_b {
        return Err(EngineError::ShapeMismatch(format!(
            "discretize_ssm: b_bar.len()={} but expected {}",
            b_bar.len(),
            expected_b
        )));
    }
    discretize_ssm_f32(a, b, delta, a_bar, b_bar, seq_len, d_inner, d_state);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selective_scan_passthrough() {
        let seq_len = 3;
        let d_inner = 2;
        let d_state = 2;

        let u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let a_bar = vec![0.0f32; seq_len * d_inner * d_state];
        let b_bar = vec![0.0f32; seq_len * d_inner * d_state];
        let c = vec![1.0f32; seq_len * d_inner * d_state];
        let d = [1.0, 1.0f32];
        let mut output = vec![0.0f32; seq_len * d_inner];

        selective_scan_f32(
            &u,
            &a_bar,
            &b_bar,
            &c,
            &d,
            &mut output,
            seq_len,
            d_inner,
            d_state,
        );
        assert_eq!(output, u);
    }

    #[test]
    fn test_selective_scan_accumulate() {
        let seq_len = 3;
        let u = [1.0, 1.0, 1.0f32];
        let a_bar = [0.5, 0.5, 0.5f32];
        let b_bar = [1.0, 1.0, 1.0f32];
        let c = [1.0, 1.0, 1.0f32];
        let d = [0.0f32];
        let mut output = [0.0f32; 3];

        selective_scan_f32(&u, &a_bar, &b_bar, &c, &d, &mut output, seq_len, 1, 1);

        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 1.5).abs() < 1e-5);
        assert!((output[2] - 1.75).abs() < 1e-5);
    }

    #[test]
    fn test_causal_conv1d_basic() {
        let input = [1.0, 2.0, 3.0, 4.0f32];
        let weight = [1.0, 0.5f32];
        let mut output = [0.0f32; 4];

        causal_conv1d_f32(&input, &weight, None, &mut output, 4, 1, 2);

        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.5).abs() < 1e-5);
        assert!((output[2] - 4.0).abs() < 1e-5);
        assert!((output[3] - 5.5).abs() < 1e-5);
    }

    #[test]
    fn test_causal_conv1d_with_bias() {
        let input = [1.0, 2.0f32];
        let weight = [1.0f32];
        let bias = [10.0f32];
        let mut output = [0.0f32; 2];

        causal_conv1d_f32(&input, &weight, Some(&bias), &mut output, 2, 1, 1);

        assert!((output[0] - 11.0).abs() < 1e-5);
        assert!((output[1] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_discretize() {
        let d_inner = 1;
        let d_state = 1;
        let seq_len = 2;

        let a = [-1.0f32];
        let b = [1.0, 1.0f32];
        let delta = [0.1, 0.2f32];
        let mut a_bar = [0.0f32; 2];
        let mut b_bar = [0.0f32; 2];

        discretize_ssm_f32(
            &a, &b, &delta, &mut a_bar, &mut b_bar, seq_len, d_inner, d_state,
        );

        assert!((a_bar[0] - (-0.1f32).exp()).abs() < 1e-5);
        assert!((b_bar[0] - 0.1).abs() < 1e-5);
        assert!((a_bar[1] - (-0.2f32).exp()).abs() < 1e-5);
        assert!((b_bar[1] - 0.2).abs() < 1e-5);
    }

    // Checked wrapper tests
    #[test]
    fn test_selective_scan_checked_valid() {
        let u = [1.0, 1.0f32];
        let abc = [0.5, 0.5f32];
        let d = [1.0f32];
        let mut output = [0.0f32; 2];
        let result = selective_scan_f32_checked(&u, &abc, &abc, &abc, &d, &mut output, 2, 1, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_selective_scan_checked_bad_u() {
        let u = [1.0f32]; // wrong size for seq_len=2, d_inner=1
        let abc = [0.5, 0.5f32];
        let d = [1.0f32];
        let mut output = [0.0f32; 2];
        let result = selective_scan_f32_checked(&u, &abc, &abc, &abc, &d, &mut output, 2, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_conv1d_checked_valid() {
        let input = [1.0, 2.0f32];
        let weight = [1.0f32];
        let mut output = [0.0f32; 2];
        let result = causal_conv1d_f32_checked(&input, &weight, None, &mut output, 2, 1, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_causal_conv1d_checked_bad_bias() {
        let input = [1.0, 2.0f32];
        let weight = [1.0f32];
        let bias = [1.0, 2.0f32]; // wrong size for d_inner=1
        let mut output = [0.0f32; 2];
        let result = causal_conv1d_f32_checked(&input, &weight, Some(&bias), &mut output, 2, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_discretize_checked_bad_a() {
        let a = [1.0, 2.0f32]; // wrong size for d_inner=1, d_state=1
        let b = [1.0f32];
        let delta = [0.1f32];
        let mut a_bar = [0.0f32; 1];
        let mut b_bar = [0.0f32; 1];
        let result = discretize_ssm_f32_checked(&a, &b, &delta, &mut a_bar, &mut b_bar, 1, 1, 1);
        assert!(result.is_err());
    }
}
