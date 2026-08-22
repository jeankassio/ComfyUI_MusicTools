#!/usr/bin/env python3
"""
Master Audio Enhancement Node
Applies professional audio mastering chain to improve Ace-Step generated audio
"""

import numpy as np
from scipy import signal

def apply_spectral_subtraction(audio, noise_profile_duration=0.5, sample_rate=44100):
    """
    Remove background noise/muddiness using spectral subtraction (OPTIMIZED)
    """
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_spectral_subtraction(audio[ch, :], noise_profile_duration, sample_rate)
        return result
    
    # Use scipy's built-in STFT for speed
    nperseg = 2048
    noverlap = nperseg // 2  # Reduced overlap for speed
    
    # Compute STFT
    frequencies, times, stft_matrix = signal.stft(
        audio, 
        fs=sample_rate, 
        nperseg=nperseg, 
        noverlap=noverlap,
        window='hann'
    )
    
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    # Estimate noise profile from first few frames (vectorized)
    noise_frames = max(1, int(noise_profile_duration * sample_rate / (nperseg - noverlap)))
    noise_frames = min(noise_frames, magnitude.shape[1] // 4)  # Max 25% of audio
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Spectral subtraction (vectorized)
    subtraction_factor = 1.5  # Reduced for more natural sound
    cleaned_magnitude = magnitude - subtraction_factor * noise_profile
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0.2 * noise_profile)  # Prevent over-subtraction
    
    # Reconstruct complex spectrum
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    
    # Inverse STFT
    _, result = signal.istft(
        cleaned_stft, 
        fs=sample_rate, 
        nperseg=nperseg, 
        noverlap=noverlap,
        window='hann'
    )
    
    # Ensure same length as input
    if len(result) > len(audio):
        result = result[:len(audio)]
    elif len(result) < len(audio):
        result = np.pad(result, (0, len(audio) - len(result)))
    
    return result.astype(np.float32)


def apply_parametric_eq(audio, low_freq=100, low_gain=0.0, mid_freq=1000, mid_gain=0.5, 
                       high_freq=5000, high_gain=1.5, sample_rate=44100):
    """
    Apply parametric EQ with 3 bands (Low, Mid, High) - FIXED VERSION
    Gains in dB - if all gains are 0, returns audio unchanged
    """
    from .utils import apply_eq

    return apply_eq(
        audio,
        [low_freq, mid_freq, high_freq],
        [low_gain, mid_gain, high_gain],
        sample_rate,
    )


def apply_multiband_compression(audio, sample_rate=44100, threshold=0.3, ratio=4.0, attack_ms=5, release_ms=50):
    """
    Fast multiband compressor with 3 bands (no slow envelope loops).
    Uses vectorized operations for speed (~5ms vs 30ms before).
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 3:
        return np.stack(
            [
                apply_multiband_compression(
                    batch, sample_rate, threshold, ratio, attack_ms, release_ms
                )
                for batch in audio
            ],
            axis=0,
        )
    if audio.ndim == 1:
        work = audio[np.newaxis, :]
        remove_channel = True
    elif audio.ndim == 2:
        work = audio
        remove_channel = False
    else:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")
    if work.shape[-1] == 0:
        return audio.copy()

    from .utils import apply_compression

    # Complementary residual bands sum back to the input exactly when no gain
    # reduction is active. The previous three independent Butterworth filters
    # produced a +3 dB crossover bump even at a 1:1 ratio.
    nyquist = sample_rate / 2.0
    low_crossover = min(250.0, nyquist * 0.15)
    high_crossover = min(3000.0, nyquist * 0.75)
    if high_crossover <= low_crossover * 1.2:
        return apply_compression(
            audio, threshold, ratio, sample_rate, attack_ms, release_ms
        )
    sos_low = signal.butter(4, low_crossover, btype="lowpass", fs=sample_rate, output="sos")
    low_band = signal.sosfilt(sos_low, work, axis=-1)
    above_low = work - low_band
    sos_mid = signal.butter(4, high_crossover, btype="lowpass", fs=sample_rate, output="sos")
    mid_band = signal.sosfilt(sos_mid, above_low, axis=-1)
    high_band = above_low - mid_band

    low_compressed = apply_compression(
        low_band, threshold * 1.1, max(1.0, ratio * 0.8), sample_rate,
        attack_ms, release_ms,
    )
    mid_compressed = apply_compression(
        mid_band, threshold, ratio, sample_rate, attack_ms, release_ms
    )
    high_compressed = apply_compression(
        high_band, threshold * 0.9, ratio * 1.2, sample_rate,
        attack_ms, release_ms,
    )

    result = low_compressed + mid_compressed + high_compressed
    result = result.astype(np.float32)
    return result[0] if remove_channel else result


def apply_loudness_normalization(audio, target_loudness=-9.0, sample_rate=44100):
    """
    Professional loudness normalization using pyloudnorm (ITU-R BS.1770-4 standard).
    Ensures broadcast-quality loudness while preserving dynamic range.
    """
    from .utils import calculate_lufs, normalize_to_lufs

    current_loudness = calculate_lufs(audio, sample_rate)
    normalized = normalize_to_lufs(audio, target_loudness, sample_rate, peak_ceiling_db=-0.5)
    if np.isfinite(current_loudness):
        print(f"[Loudness] {current_loudness:.1f} LUFS -> {target_loudness:.1f} LUFS")
    return normalized.astype(np.float32)


def apply_clarity_enhancement(audio, clarity_amount=0.5, sample_rate=44100):
    """
    Professional clarity enhancement: harmonic exciter + transient shaper + presence boost.
    Adds air, detail, and punch for studio-quality definition.
    """
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_clarity_enhancement(audio[ch, :], clarity_amount, sample_rate)
        return result
    
    if clarity_amount < 0.01:
        return audio.astype(np.float32)
    if audio.size == 0:
        return audio.astype(np.float32)

    result = audio.copy()
    nyquist = sample_rate / 2.0

    def band_component(low, high, order=2):
        high = min(float(high), nyquist * 0.95)
        low = max(float(low), 1.0)
        if low >= high:
            return np.zeros_like(audio, dtype=np.float32)
        sos = signal.butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        return signal.sosfilt(sos, audio).astype(np.float32)

    def high_component(cutoff, order=2):
        cutoff = float(cutoff)
        if cutoff >= nyquist * 0.95:
            return np.zeros_like(audio, dtype=np.float32)
        sos = signal.butter(order, cutoff, btype="highpass", fs=sample_rate, output="sos")
        return signal.sosfilt(sos, audio).astype(np.float32)
    
    # 1. Transient shaper: enhance attack for punchier sound
    diff = np.diff(audio, prepend=audio[0])
    transients = np.abs(diff)
    transient_threshold = np.percentile(transients, 90)
    transient_mask = transients > transient_threshold
    transient_boost = np.zeros_like(audio)
    transient_boost[transient_mask] = diff[transient_mask] * 0.3 * clarity_amount
    result = result + transient_boost
    
    # 2. Harmonic exciter: add even harmonics for warmth and detail (8-12 kHz)
    high_band = band_component(8000, 12000, order=4)
    # Soft saturation to generate harmonics
    excited = np.tanh(high_band * 2.5) * 0.4
    result = result + excited * clarity_amount * 0.4
    
    # 3. Presence boost (2-5 kHz) for vocal/instrument clarity
    presence = band_component(2000, 5000, order=3)
    result = result + presence * clarity_amount * 0.25
    
    # 4. Air band (10-16 kHz) for sparkle and openness
    air = high_component(10000, order=2)
    result = result + air * clarity_amount * 0.15
    
    # 5. Mud removal (80-200 Hz cut)
    mud = band_component(80, 200, order=2)
    result = result - mud * clarity_amount * 0.2
    
    return result.astype(np.float32)


def apply_soft_limiter(audio, threshold=0.98, lookahead_ms=2.0, sample_rate=44100):
    """
    Fast true-peak limiter with simple lookahead (~5ms vs 50ms before).
    Prevents clipping while maintaining speed.
    """
    from .limiter import apply_true_peak_limiter

    threshold = float(np.clip(threshold, 1e-6, 0.9999))
    ceiling_db = 20.0 * np.log10(threshold)
    limited, _, _ = apply_true_peak_limiter(
        audio,
        sample_rate=sample_rate,
        ceiling_db=ceiling_db,
        lookahead_ms=lookahead_ms,
        release_ms=80.0,
        oversample=4,
    )
    return limited.astype(np.float32)
