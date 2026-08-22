#!/usr/bin/env python3
"""
Vocal enhancement tools for professional vocal processing.
De-esser, breath smoother, and intelligent reverb for natural-sounding vocals.
"""

import numpy as np
from scipy import signal


def apply_deesser(audio, sample_rate=44100, sensitivity=0.5):
    """
    Fast de-esser to reduce sibilance (4-8 kHz hiss).
    Uses parametric notch filter for speed (2-3ms vs 50ms spectral).
    
    Args:
        audio: Audio array (channels, samples) or (samples,)
        sample_rate: Sample rate
        sensitivity: 0-1, how aggressive the de-esser is (0.5 is balanced)
    
    Returns:
        De-essed audio
    """
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_deesser(audio[ch, :], sample_rate, sensitivity)
        return result
    
    audio = audio.astype(np.float32)
    if sensitivity < 0.01:
        return audio
    
    # Fast parametric notch filter on sibilant band (5-7 kHz peak)
    # Attenuate 4-8 kHz band with steeper notch
    center_freq = min(5500 + (sensitivity * 1000), sample_rate * 0.475)
    if center_freq <= 100.0:
        return audio
    Q_factor = 1.5 + sensitivity * 1.5  # Steeper notch for higher sensitivity
    gain_db = -(3.0 + sensitivity * 5)  # -3 to -8 dB reduction
    
    # Design peaking EQ (notch) filter
    w0 = 2 * np.pi * center_freq / sample_rate
    alpha = np.sin(w0) / (2 * Q_factor)
    
    # Convert gain_db to linear
    A = 10 ** (gain_db / 40)
    
    # Peaking filter coefficients
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 + alpha / A], dtype=np.float32)
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 + alpha * A], dtype=np.float32)
    
    # Apply filter (single pass - very fast)
    result = signal.lfilter(b, a, audio)
    
    return result.astype(np.float32)


def apply_breath_smoother(audio, sample_rate=44100, aggression=0.5):
    """
    Fast breath smoother using gentle low-pass filter.
    Smooths breath artifacts by reducing sudden envelope changes (1-2ms vs 20ms envelope detection).
    
    Args:
        audio: Audio array (channels, samples) or (samples,)
        sample_rate: Sample rate
        aggression: 0-1, how much to smooth (0.5 is balanced)
    
    Returns:
        Smoothed audio
    """
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_breath_smoother(audio[ch, :], sample_rate, aggression)
        return result
    
    audio = audio.astype(np.float32)
    if aggression < 0.01:
        return audio
    
    # Fast approach: light low-pass filter on breath/pop frequencies (100-300 Hz)
    # Higher aggression = lower cutoff = more smoothing
    cutoff_freq = 300 - aggression * 150  # 150-300 Hz range
    
    # Apply gentle low-pass with minimal phase distortion
    sos_smooth = signal.butter(2, cutoff_freq, btype='low', fs=sample_rate, output='sos')
    smoothed = signal.sosfilt(sos_smooth, audio).astype(np.float32)
    
    # Blend: keep original transients but smooth artifacts
    # Higher aggression = more blending
    blend = 0.3 * aggression
    result = audio * (1.0 - blend) + smoothed * blend
    
    return result.astype(np.float32)


def apply_vocal_reverb(audio, sample_rate=44100, reverb_amount=0.3, reverb_type="small_room"):
    """
    Fast multi-tap delay reverb for vocals (5-10ms vs 50ms Schroeder).
    Creates natural reverb space using delayed copies with decay.
    
    Args:
        audio: Audio array (channels, samples) or (samples,)
        sample_rate: Sample rate
        reverb_amount: 0-1, mix of reverb
        reverb_type: "small_room", "medium_room", "large_hall", "plate"
    
    Returns:
        Reverb-processed audio
    """
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_vocal_reverb(audio[ch, :], sample_rate, reverb_amount, reverb_type)
        return result
    
    audio = audio.astype(np.float32)
    if reverb_amount < 0.01:
        return audio
    
    # Fast multi-tap delay reverb parameters
    reverb_params = {
        "small_room": {"delays_ms": [20, 35, 55, 75], "gains": [0.5, 0.35, 0.2, 0.1]},
        "medium_room": {"delays_ms": [30, 50, 75, 100], "gains": [0.55, 0.4, 0.25, 0.15]},
        "large_hall": {"delays_ms": [50, 80, 120, 160], "gains": [0.6, 0.45, 0.3, 0.2]},
        "plate": {"delays_ms": [25, 40, 60, 85], "gains": [0.6, 0.4, 0.25, 0.15]},
    }
    
    params = reverb_params.get(reverb_type, reverb_params["small_room"])
    
    # Create multi-tap delays
    reverb_signal = np.zeros_like(audio)
    
    for delay_ms, gain in zip(params["delays_ms"], params["gains"]):
        delay_samples = int(delay_ms * sample_rate / 1000)
        if delay_samples > 0 and delay_samples < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * gain
            reverb_signal += delayed
    
    # Gentle high-frequency damping for natural reverb
    damping_cutoff = min(8000.0, sample_rate * 0.475)
    sos_damp = signal.butter(1, damping_cutoff, btype='low', fs=sample_rate, output='sos')
    reverb_signal = signal.sosfilt(sos_damp, reverb_signal)
    
    # Mix original with reverb
    mix = np.clip(reverb_amount, 0, 1)
    result = audio * (1 - mix) + reverb_signal * mix
    
    return result.astype(np.float32)


def apply_vocal_naturalizer(audio, sample_rate=44100, amount=0.5):
    """
    Soften metallic high-frequency and abrupt waveform artifacts in AI vocals.

    This is a deterministic spectral smoother; it does not claim to reconstruct
    an original pitch contour or undo pitch correction.
    
    Args:
        audio: Audio array (channels, samples) or (samples,)
        sample_rate: Sample rate
        amount: 0-1, how much to naturalize (0.5 is balanced)
    
    Returns:
        Naturalized audio
    """
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_vocal_naturalizer(audio[ch, :], sample_rate, amount)
        return result
    
    audio = audio.astype(np.float32)
    if amount < 0.01:
        return audio
    
    if audio.size < 2:
        return audio.copy()

    result = audio.copy()
    nyquist = sample_rate / 2.0

    # Reduce the metallic 6-10 kHz band when the source sample rate contains it.
    metal_low = 6000.0
    metal_high = min(10000.0, nyquist * 0.95)
    if metal_low < metal_high:
        sos_metal = signal.butter(
            3, [metal_low, metal_high], btype="bandpass", fs=sample_rate, output="sos"
        )
        metallic = signal.sosfilt(sos_metal, audio).astype(np.float32)
        result -= metallic * (0.3 * amount)

    # Smooth only a small portion of abrupt sample-to-sample changes. This is
    # intentionally subtle so consonants and drum transients in a full mix are
    # not erased.
    diff = np.diff(result, prepend=result[0])
    sos_smooth = signal.butter(1, 50, btype='low', fs=sample_rate, output='sos')
    smoothed_diff = signal.sosfilt(sos_smooth, diff).astype(np.float32)
    smooth_blend = 0.15 * amount
    result = result - diff * smooth_blend + smoothed_diff * smooth_blend
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(result))
    if max_val > 0.95:
        result = result * (0.95 / max_val)
    
    return result.astype(np.float32)
