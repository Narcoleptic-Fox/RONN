//! GPU softmax kernel.

use cubecl::prelude::*;

/// In-place softmax over a row of `seq_len` elements.
///
/// Numerically stable: max-subtract, exp, sum, divide.
/// Launch one cube per row, one thread per cube (serial per row — correct first).
#[cube(launch)]
pub fn softmax_kernel(data: &mut Array<f32>, row_offset: u32, seq_len: u32) {
    let tid = UNIT_POS_X;

    if tid == 0u32 {
        // Find max.
        let mut max_val = data[row_offset];
        for i in 1..seq_len {
            let val = data[row_offset + i];
            if val > max_val {
                max_val = val;
            }
        }

        // Exp and sum.
        let mut sum = 0.0f32;
        for i in 0..seq_len {
            let exp_val = f32::exp(data[row_offset + i] - max_val);
            data[row_offset + i] = exp_val;
            sum += exp_val;
        }

        // Normalize.
        for i in 0..seq_len {
            data[row_offset + i] = data[row_offset + i] / sum;
        }
    }
}
