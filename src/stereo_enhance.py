#!/usr/bin/env python3
"""
Advanced stereo imaging and enhancement for professional mastering.
"""

import numpy as np
from scipy import signal


def apply_stereo_widening(audio, width=1.5, sample_rate=44100):
    """
    Professional stereo width enhancement using mid/side processing.
    
    Args:
        audio: Stereo audio (2, samples)
        width: 0.0 = mono, 1.0 = original, 2.0 = maximum width
        sample_rate: Sample rate
    
    Returns:
        Widened stereo audio
    """
    if len(audio.shape) != 2 or audio.shape[0] != 2:
        return audio  # Not stereo
    
    left = audio[0, :]
    right = audio[1, :]
    
    # Convert to mid/side
    mid = (left + right) / 2
    side = (left - right) / 2
    
    # Enhance side signal with frequency-dependent width
    # More width in high frequencies (natural stereo perception)
    sos_low = signal.butter(2, 200, btype='low', fs=sample_rate, output='sos')
    sos_high = signal.butter(2, 200, btype='high', fs=sample_rate, output='sos')
    
    side_low = signal.sosfilt(sos_low, side) * np.clip(width * 0.8, 0, 2)
    side_high = signal.sosfilt(sos_high, side) * np.clip(width * 1.2, 0, 2)
    side_enhanced = side_low + side_high
    
    # Convert back to left/right
    left_out = mid + side_enhanced
    right_out = mid - side_enhanced
    
    # Prevent clipping
    max_val = max(np.abs(left_out).max(), np.abs(right_out).max())
    if max_val > 1.0:
        left_out = left_out / max_val
        right_out = right_out / max_val
    
    return np.stack([left_out, right_out], axis=0).astype(np.float32)


def apply_stereo_correlation_fix(audio, sample_rate=44100):
    """
    Fix stereo correlation issues (phase problems) that cause mono compatibility issues.
    
    Args:
        audio: Stereo audio (2, samples)
        sample_rate: Sample rate
    
    Returns:
        Phase-corrected stereo audio
    """
    if len(audio.shape) != 2 or audio.shape[0] != 2:
        return audio
    
    left = audio[0, :]
    right = audio[1, :]
    
    # Calculate correlation
    correlation = np.corrcoef(left, right)[0, 1]
    
    # If correlation is very negative, we have phase issues
    if correlation < -0.3:
        # Invert one channel to fix phase
        right = -right
        print(f"[Stereo] Fixed phase inversion (corr: {correlation:.2f})")
    
    # Mid/side processing to ensure mono compatibility
    mid = (left + right) / 2
    side = (left - right) / 2
    
    # Slightly reduce side in low frequencies for better mono compatibility
    sos_low = signal.butter(2, 150, btype='low', fs=sample_rate, output='sos')
    sos_high = signal.butter(2, 150, btype='high', fs=sample_rate, output='sos')
    
    side_low = signal.sosfilt(sos_low, side) * 0.7
    side_high = signal.sosfilt(sos_high, side)
    side_fixed = side_low + side_high
    
    left_out = mid + side_fixed
    right_out = mid - side_fixed
    
    return np.stack([left_out, right_out], axis=0).astype(np.float32)


def apply_pseudo_stereo(audio, amount=0.5, sample_rate=44100):
    """
    Convert mono to pseudo-stereo or enhance narrow stereo.
    Uses Haas effect and decorrelation for natural stereo width.
    
    Args:
        audio: Mono or stereo audio
        amount: 0-1, strength of stereo effect
        sample_rate: Sample rate
    
    Returns:
        Stereo audio
    """
    if len(audio.shape) == 1:
        # Mono input
        mono = audio
    elif audio.shape[0] == 1:
        mono = audio[0, :]
    else:
        # Already stereo, enhance it
        return apply_stereo_widening(audio, 1.0 + amount * 0.5, sample_rate)
    
    # Create decorrelated version using allpass filters
    # This creates natural stereo effect
    def allpass_delay(signal_in, delay_samples, decay=0.7):
        delayed = np.zeros_like(signal_in)
        delay_samples = int(delay_samples)
        if delay_samples > 0 and delay_samples < len(signal_in):
            delayed[delay_samples:] = signal_in[:-delay_samples] * decay
        return signal_in + delayed
    
    # Left: original
    left = mono.copy()
    
    # Right: slight delay + decorrelation
    delay_ms = 15 * amount  # 0-15ms delay
    delay_samples = int(delay_ms * sample_rate / 1000)
    right = allpass_delay(mono, delay_samples, 0.6)
    
    # Add complementary filtering for naturalness
    sos_left = signal.butter(1, [500, 4000], btype='band', fs=sample_rate, output='sos')
    sos_right = signal.butter(1, [600, 3500], btype='band', fs=sample_rate, output='sos')
    
    left_color = signal.sosfilt(sos_left, mono) * amount * 0.1
    right_color = signal.sosfilt(sos_right, mono) * amount * 0.1
    
    left = left + left_color
    right = right + right_color
    
    # Normalize
    max_val = max(np.abs(left).max(), np.abs(right).max())
    if max_val > 1.0:
        left = left / max_val
        right = right / max_val
    
    return np.stack([left, right], axis=0).astype(np.float32)
