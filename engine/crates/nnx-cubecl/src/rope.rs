//! GPU rotary position encoding (RoPE) kernel.

use cubecl::prelude::*;

/// Apply rotary position encoding in-place to a head vector.
///
/// RoPE rotates pairs of dimensions: for pair (2i, 2i+1),
/// ```text
/// x[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
/// x[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
/// ```
/// where `θ_i = position / (freq_base ^ (2i / head_dim))`.
///
/// Launch one cube per head, one thread per cube (serial — correct first).
#[cube(launch)]
pub fn rope_kernel(
    data: &mut Array<f32>,
    head_offset: u32,
    head_dim: u32,
    position: u32,
    freq_base: f32,
) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        let half_dim = head_dim / 2u32;
        for pair in 0..half_dim {
            let dim_fraction = f32::cast_from(2u32 * pair) / f32::cast_from(head_dim);
            let freq = 1.0f32 / f32::powf(freq_base, dim_fraction);
            let angle = f32::cast_from(position) * freq;
            let cos_val = f32::cos(angle);
            let sin_val = f32::sin(angle);

            let idx_even = head_offset + 2u32 * pair;
            let idx_odd = idx_even + 1u32;

            let x0 = data[idx_even];
            let x1 = data[idx_odd];

            data[idx_even] = x0 * cos_val - x1 * sin_val;
            data[idx_odd] = x0 * sin_val + x1 * cos_val;
        }
    }
}
