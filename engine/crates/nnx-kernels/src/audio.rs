//! Audio preprocessing operations: STFT, MelFilterBank.
//!
//! Used by Whisper, Wav2Vec2, and other audio models.

use std::f32::consts::PI;

use nnx_core::error::EngineError;

/// Short-Time Fourier Transform (STFT).
///
/// Splits the signal into overlapping frames, applies a window function,
/// and computes the DFT of each frame.
///
/// `signal`: input audio samples [signal_len]
/// `output_real`: real part of STFT output [num_frames, n_fft/2 + 1]
/// `output_imag`: imaginary part of STFT output [num_frames, n_fft/2 + 1]
/// `window`: window function coefficients [n_fft] (e.g., Hann window)
/// `n_fft`: FFT size
/// `hop_length`: stride between frames
///
/// Returns the number of frames computed.
pub fn stft_f32(
    signal: &[f32],
    output_real: &mut [f32],
    output_imag: &mut [f32],
    window: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> usize {
    assert_eq!(window.len(), n_fft);

    let freq_bins = n_fft / 2 + 1;
    let num_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    assert!(output_real.len() >= num_frames * freq_bins);
    assert!(output_imag.len() >= num_frames * freq_bins);

    for frame in 0..num_frames {
        let start = frame * hop_length;

        // DFT of windowed frame (only positive frequencies)
        for k in 0..freq_bins {
            let mut re = 0.0f32;
            let mut im = 0.0f32;

            for n in 0..n_fft {
                let sample = if start + n < signal.len() {
                    signal[start + n] * window[n]
                } else {
                    0.0
                };
                let angle = -2.0 * PI * k as f32 * n as f32 / n_fft as f32;
                re += sample * angle.cos();
                im += sample * angle.sin();
            }

            output_real[frame * freq_bins + k] = re;
            output_imag[frame * freq_bins + k] = im;
        }
    }

    num_frames
}

/// Checked version of `stft_f32` that validates dimensions before computing.
pub fn stft_f32_checked(
    signal: &[f32],
    output_real: &mut [f32],
    output_imag: &mut [f32],
    window: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> nnx_core::error::Result<usize> {
    if window.len() != n_fft {
        return Err(EngineError::ShapeMismatch(
            format!("stft: window.len()={} but n_fft={}", window.len(), n_fft)
        ));
    }
    if hop_length == 0 {
        return Err(EngineError::Kernel(
            "stft: hop_length must be non-zero".to_string()
        ));
    }
    let freq_bins = n_fft / 2 + 1;
    let num_frames = if signal.len() >= n_fft {
        (signal.len() - n_fft) / hop_length + 1
    } else {
        0
    };
    if output_real.len() < num_frames * freq_bins {
        return Err(EngineError::ShapeMismatch(
            format!("stft: output_real.len()={} but need at least {}", output_real.len(), num_frames * freq_bins)
        ));
    }
    if output_imag.len() < num_frames * freq_bins {
        return Err(EngineError::ShapeMismatch(
            format!("stft: output_imag.len()={} but need at least {}", output_imag.len(), num_frames * freq_bins)
        ));
    }
    Ok(stft_f32(signal, output_real, output_imag, window, n_fft, hop_length))
}

/// Compute the magnitude spectrogram from STFT output.
///
/// `real`, `imag`: STFT outputs [num_frames, freq_bins]
/// `output`: magnitude spectrogram [num_frames, freq_bins]
pub fn stft_magnitude_f32(real: &[f32], imag: &[f32], output: &mut [f32]) {
    assert_eq!(real.len(), imag.len());
    assert_eq!(real.len(), output.len());
    for i in 0..real.len() {
        output[i] = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
    }
}

/// Checked version of `stft_magnitude_f32`.
pub fn stft_magnitude_f32_checked(real: &[f32], imag: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    if real.len() != imag.len() {
        return Err(EngineError::ShapeMismatch(
            format!("stft_magnitude: real.len()={} != imag.len()={}", real.len(), imag.len())
        ));
    }
    if real.len() != output.len() {
        return Err(EngineError::ShapeMismatch(
            format!("stft_magnitude: real.len()={} != output.len()={}", real.len(), output.len())
        ));
    }
    stft_magnitude_f32(real, imag, output);
    Ok(())
}

/// Compute power spectrogram (magnitude squared).
pub fn stft_power_f32(real: &[f32], imag: &[f32], output: &mut [f32]) {
    assert_eq!(real.len(), imag.len());
    assert_eq!(real.len(), output.len());
    for i in 0..real.len() {
        output[i] = real[i] * real[i] + imag[i] * imag[i];
    }
}

/// Checked version of `stft_power_f32`.
pub fn stft_power_f32_checked(real: &[f32], imag: &[f32], output: &mut [f32]) -> nnx_core::error::Result<()> {
    if real.len() != imag.len() {
        return Err(EngineError::ShapeMismatch(
            format!("stft_power: real.len()={} != imag.len()={}", real.len(), imag.len())
        ));
    }
    if real.len() != output.len() {
        return Err(EngineError::ShapeMismatch(
            format!("stft_power: real.len()={} != output.len()={}", real.len(), output.len())
        ));
    }
    stft_power_f32(real, imag, output);
    Ok(())
}

/// Generate a Hann window of the given size.
pub fn hann_window(output: &mut [f32]) {
    let n = output.len();
    for i in 0..n {
        output[i] = 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos());
    }
}

/// Convert frequency in Hz to mel scale.
#[inline]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
#[inline]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Generate a mel filterbank matrix.
///
/// `output`: [n_mels, n_fft/2 + 1] -- each row is a triangular filter
/// `n_mels`: number of mel bands
/// `n_fft`: FFT size
/// `sample_rate`: audio sample rate in Hz
/// `f_min`: minimum frequency in Hz
/// `f_max`: maximum frequency in Hz
pub fn mel_filterbank_f32(
    output: &mut [f32],
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) {
    let freq_bins = n_fft / 2 + 1;
    assert_eq!(output.len(), n_mels * freq_bins);

    // Mel points: n_mels + 2 points from f_min to f_max in mel scale
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mut mel_points = vec![0.0f32; n_mels + 2];
    for i in 0..n_mels + 2 {
        let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
        mel_points[i] = mel_to_hz(mel);
    }

    // Convert Hz points to FFT bin indices
    let mut bin_points = vec![0.0f32; n_mels + 2];
    for i in 0..n_mels + 2 {
        bin_points[i] = mel_points[i] * (n_fft as f32 + 1.0) / sample_rate;
    }

    output.fill(0.0);

    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for k in 0..freq_bins {
            let kf = k as f32;
            let weight = if kf >= f_left && kf <= f_center {
                // Rising slope
                if (f_center - f_left).abs() < f32::EPSILON {
                    0.0
                } else {
                    (kf - f_left) / (f_center - f_left)
                }
            } else if kf > f_center && kf <= f_right {
                // Falling slope
                if (f_right - f_center).abs() < f32::EPSILON {
                    0.0
                } else {
                    (f_right - kf) / (f_right - f_center)
                }
            } else {
                0.0
            };

            output[m * freq_bins + k] = weight;
        }
    }
}

/// Checked version of `mel_filterbank_f32`.
pub fn mel_filterbank_f32_checked(
    output: &mut [f32],
    n_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> nnx_core::error::Result<()> {
    let freq_bins = n_fft / 2 + 1;
    if output.len() != n_mels * freq_bins {
        return Err(EngineError::ShapeMismatch(
            format!("mel_filterbank: output.len()={} but n_mels*freq_bins={}", output.len(), n_mels * freq_bins)
        ));
    }
    if sample_rate <= 0.0 {
        return Err(EngineError::Kernel(
            format!("mel_filterbank: sample_rate must be positive, got {}", sample_rate)
        ));
    }
    mel_filterbank_f32(output, n_mels, n_fft, sample_rate, f_min, f_max);
    Ok(())
}

/// Apply mel filterbank to a power/magnitude spectrogram.
///
/// `filterbank`: [n_mels, freq_bins]
/// `spectrogram`: [num_frames, freq_bins]
/// `output`: [num_frames, n_mels]
///
/// output = spectrogram @ filterbank^T
pub fn apply_mel_filterbank_f32(
    filterbank: &[f32],
    spectrogram: &[f32],
    output: &mut [f32],
    num_frames: usize,
    n_mels: usize,
    freq_bins: usize,
) {
    assert_eq!(filterbank.len(), n_mels * freq_bins);
    assert_eq!(spectrogram.len(), num_frames * freq_bins);
    assert_eq!(output.len(), num_frames * n_mels);

    for f in 0..num_frames {
        for m in 0..n_mels {
            let mut sum = 0.0f32;
            for k in 0..freq_bins {
                sum += spectrogram[f * freq_bins + k] * filterbank[m * freq_bins + k];
            }
            output[f * n_mels + m] = sum;
        }
    }
}

/// Checked version of `apply_mel_filterbank_f32`.
pub fn apply_mel_filterbank_f32_checked(
    filterbank: &[f32],
    spectrogram: &[f32],
    output: &mut [f32],
    num_frames: usize,
    n_mels: usize,
    freq_bins: usize,
) -> nnx_core::error::Result<()> {
    if filterbank.len() != n_mels * freq_bins {
        return Err(EngineError::ShapeMismatch(
            format!("apply_mel_filterbank: filterbank.len()={} but n_mels*freq_bins={}", filterbank.len(), n_mels * freq_bins)
        ));
    }
    if spectrogram.len() != num_frames * freq_bins {
        return Err(EngineError::ShapeMismatch(
            format!("apply_mel_filterbank: spectrogram.len()={} but num_frames*freq_bins={}", spectrogram.len(), num_frames * freq_bins)
        ));
    }
    if output.len() != num_frames * n_mels {
        return Err(EngineError::ShapeMismatch(
            format!("apply_mel_filterbank: output.len()={} but num_frames*n_mels={}", output.len(), num_frames * n_mels)
        ));
    }
    apply_mel_filterbank_f32(filterbank, spectrogram, output, num_frames, n_mels, freq_bins);
    Ok(())
}

/// Log mel spectrogram: log(max(mel_spec, eps)).
pub fn log_mel_f32(x: &[f32], output: &mut [f32], eps: f32) {
    assert_eq!(x.len(), output.len());
    for i in 0..x.len() {
        output[i] = x[i].max(eps).ln();
    }
}

/// Checked version of `log_mel_f32`.
pub fn log_mel_f32_checked(x: &[f32], output: &mut [f32], eps: f32) -> nnx_core::error::Result<()> {
    if x.len() != output.len() {
        return Err(EngineError::ShapeMismatch(
            format!("log_mel: x.len()={} != output.len()={}", x.len(), output.len())
        ));
    }
    log_mel_f32(x, output, eps);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let mut w = [0.0f32; 8];
        hann_window(&mut w);
        assert!((w[0] - 0.0).abs() < 1e-5);
        assert!(w[1] > 0.0);
        assert!((w[4] - 1.0).abs() < 1e-5);
        assert!((w[1] - w[7]).abs() < 1e-5);
        assert!((w[2] - w[6]).abs() < 1e-5);
        assert!((w[3] - w[5]).abs() < 1e-5);
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        for &hz in &[0.0, 100.0, 440.0, 1000.0, 8000.0, 16000.0] {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            assert!((back - hz).abs() < 0.01, "roundtrip failed for {hz}Hz");
        }
    }

    #[test]
    fn test_stft_basic() {
        let n_fft = 8;
        let hop = 4;
        let signal = vec![1.0f32; 16];
        let mut window = vec![0.0f32; n_fft];
        hann_window(&mut window);

        let freq_bins = n_fft / 2 + 1;
        let max_frames = (signal.len() - n_fft) / hop + 1;
        let mut real = vec![0.0f32; max_frames * freq_bins];
        let mut imag = vec![0.0f32; max_frames * freq_bins];

        let num_frames = stft_f32(&signal, &mut real, &mut imag, &window, n_fft, hop);
        assert!(num_frames > 0);

        let mut mag = vec![0.0f32; num_frames * freq_bins];
        stft_magnitude_f32(
            &real[..num_frames * freq_bins],
            &imag[..num_frames * freq_bins],
            &mut mag,
        );

        for f in 0..num_frames {
            let dc_mag = mag[f * freq_bins];
            for k in 1..freq_bins {
                assert!(
                    dc_mag >= mag[f * freq_bins + k] - 1e-4,
                    "DC should be largest: dc={dc_mag}, bin[{k}]={}",
                    mag[f * freq_bins + k]
                );
            }
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let n_mels = 40;
        let n_fft = 512;
        let freq_bins = n_fft / 2 + 1;
        let mut fb = vec![0.0f32; n_mels * freq_bins];
        mel_filterbank_f32(&mut fb, n_mels, n_fft, 16000.0, 0.0, 8000.0);

        for &v in &fb {
            assert!(v >= 0.0);
        }

        let total: f32 = fb.iter().sum();
        assert!(total > 0.0, "filterbank should have non-zero weights");
    }

    #[test]
    fn test_log_mel() {
        let x = [1.0, 0.0001, 100.0f32];
        let mut out = [0.0f32; 3];
        log_mel_f32(&x, &mut out, 1e-10);
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!(out[1] < 0.0);
        assert!(out[2] > 0.0);
    }

    // Checked wrapper tests
    #[test]
    fn test_stft_checked_valid() {
        let n_fft = 8;
        let hop = 4;
        let signal = vec![1.0f32; 16];
        let mut window = vec![0.0f32; n_fft];
        hann_window(&mut window);

        let freq_bins = n_fft / 2 + 1;
        let max_frames = (signal.len() - n_fft) / hop + 1;
        let mut real = vec![0.0f32; max_frames * freq_bins];
        let mut imag = vec![0.0f32; max_frames * freq_bins];

        let result = stft_f32_checked(&signal, &mut real, &mut imag, &window, n_fft, hop);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0);
    }

    #[test]
    fn test_stft_checked_bad_window() {
        let signal = vec![1.0f32; 16];
        let window = vec![0.0f32; 4]; // wrong size for n_fft=8
        let mut real = vec![0.0f32; 20];
        let mut imag = vec![0.0f32; 20];

        let result = stft_f32_checked(&signal, &mut real, &mut imag, &window, 8, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_stft_checked_zero_hop() {
        let signal = vec![1.0f32; 16];
        let window = vec![0.0f32; 8];
        let mut real = vec![0.0f32; 20];
        let mut imag = vec![0.0f32; 20];

        let result = stft_f32_checked(&signal, &mut real, &mut imag, &window, 8, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_stft_checked_output_too_small() {
        let signal = vec![1.0f32; 16];
        let mut window = vec![0.0f32; 8];
        hann_window(&mut window);
        let mut real = vec![0.0f32; 2]; // too small
        let mut imag = vec![0.0f32; 2];

        let result = stft_f32_checked(&signal, &mut real, &mut imag, &window, 8, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_log_mel_checked_mismatch() {
        let x = [1.0, 2.0f32];
        let mut out = [0.0f32; 3]; // wrong size
        let result = log_mel_f32_checked(&x, &mut out, 1e-10);
        assert!(result.is_err());
    }
}
