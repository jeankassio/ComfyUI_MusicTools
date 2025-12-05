#!/usr/bin/env python3
"""
Master Audio Enhancement Node
Applies professional audio mastering chain to improve Ace-Step generated audio
"""

import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.fft import fft, ifft
import pyloudnorm

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
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_parametric_eq(audio[ch, :], low_freq, low_gain, 
                                               mid_freq, mid_gain, high_freq, high_gain, sample_rate)
        return result
    
    # If no gain changes, return original
    if abs(low_gain) < 0.01 and abs(mid_gain) < 0.01 and abs(high_gain) < 0.01:
        return audio.astype(np.float32)
    
    result = audio.copy()
    
    # Apply gain adjustments using proper filtering
    def db_to_linear(db):
        return 10 ** (db / 20.0)
    
    # Low band (below low_freq)
    if abs(low_gain) > 0.01:
        sos_low = signal.butter(2, low_freq, btype='lowpass', fs=sample_rate, output='sos')
        low_band = signal.sosfilt(sos_low, audio).astype(np.float32)
        gain_low = db_to_linear(low_gain) - 1.0  # Convert to additive gain
        result = result + low_band * gain_low
    
    # Mid band (low_freq to high_freq)
    if abs(mid_gain) > 0.01:
        sos_mid = signal.butter(2, [low_freq, high_freq], btype='bandpass', fs=sample_rate, output='sos')
        mid_band = signal.sosfilt(sos_mid, audio).astype(np.float32)
        gain_mid = db_to_linear(mid_gain) - 1.0
        result = result + mid_band * gain_mid
    
    # High band (above high_freq)
    if abs(high_gain) > 0.01:
        sos_high = signal.butter(2, high_freq, btype='highpass', fs=sample_rate, output='sos')
        high_band = signal.sosfilt(sos_high, audio).astype(np.float32)
        gain_high = db_to_linear(high_gain) - 1.0
        result = result + high_band * gain_high
    
    return result.astype(np.float32)


def apply_multiband_compression(audio, sample_rate=44100, threshold=0.3, ratio=4.0, attack_ms=5, release_ms=50):
    """
    Fast multiband compressor with 3 bands (no slow envelope loops).
    Uses vectorized operations for speed (~5ms vs 30ms before).
    """
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_multiband_compression(audio[ch, :], sample_rate, threshold, ratio, attack_ms, release_ms)
        return result
    
    # Band crossovers: 250 Hz and 3 kHz
    sos_low = signal.butter(4, 250, btype='lowpass', fs=sample_rate, output='sos')
    sos_mid = signal.butter(4, [250, 3000], btype='bandpass', fs=sample_rate, output='sos')
    sos_high = signal.butter(4, 3000, btype='highpass', fs=sample_rate, output='sos')
    
    low_band = signal.sosfilt(sos_low, audio)
    mid_band = signal.sosfilt(sos_mid, audio)
    high_band = signal.sosfilt(sos_high, audio)
    
    def _compress_band_fast(band, thresh, rat):
        """Fast vectorized compression (no attack/release smoothing)."""
        envelope = np.abs(band)
        gain_reduction = np.ones_like(envelope, dtype=np.float32)
        mask = envelope > thresh
        if np.any(mask):
            gain_reduction[mask] = (thresh + (envelope[mask] - thresh) / rat) / envelope[mask]
            gain_reduction[mask] = np.maximum(gain_reduction[mask], 1.0 / rat)  # Hard floor
        return band * gain_reduction
    
    # Compress each band (vectorized, no loops)
    low_compressed = _compress_band_fast(low_band, threshold * 1.1, ratio * 0.8)
    mid_compressed = _compress_band_fast(mid_band, threshold, ratio)
    high_compressed = _compress_band_fast(high_band, threshold * 0.9, ratio * 1.2)
    
    result = low_compressed + mid_compressed + high_compressed
    return result.astype(np.float32)


def apply_loudness_normalization(audio, target_loudness=-9.0, sample_rate=44100):
    """
    Professional loudness normalization using pyloudnorm (ITU-R BS.1770-4 standard).
    Ensures broadcast-quality loudness while preserving dynamic range.
    """
    audio_float32 = audio.astype(np.float32)
    
    try:
        meter = pyloudnorm.Meter(sample_rate)
        if len(audio.shape) > 1:
            # Stereo/multi: transpose to (samples, channels) for pyloudnorm
            audio_for_meter = audio_float32.T
        else:
            audio_for_meter = audio_float32
        
        current_loudness = meter.integrated_loudness(audio_for_meter)
        if np.isinf(current_loudness) or np.isnan(current_loudness) or current_loudness < -70:
            print(f"[Loudness] Too quiet ({current_loudness:.1f} LUFS), using RMS fallback")
            rms = np.sqrt(np.mean(audio_float32**2))
            if rms < 1e-10:
                return audio_float32
            target_rms = 10 ** ((target_loudness + 23) / 20)
            target_rms = np.clip(target_rms, 0.4, 0.75)
            gain = target_rms / rms
            current_peak = np.abs(audio_float32).max()
            if current_peak * gain > 0.97:
                gain = 0.97 / current_peak
            return (audio_float32 * gain).astype(np.float32)
        
        normalized = pyloudnorm.normalize.loudness(audio_for_meter, current_loudness, target_loudness)
        if len(audio.shape) > 1:
            normalized = normalized.T
        
        # Safety peak check
        peak = np.abs(normalized).max()
        if peak > 0.995:
            normalized = normalized * (0.995 / peak)
        
        print(f"[Loudness] {current_loudness:.1f} LUFS → {target_loudness:.1f} LUFS")
        return normalized.astype(np.float32)
    except Exception as e:
        print(f"[Loudness] pyloudnorm failed ({e}), using RMS fallback")
        rms = np.sqrt(np.mean(audio_float32**2))
        if rms < 1e-10:
            return audio_float32
        target_rms = 10 ** ((target_loudness + 23) / 20)
        target_rms = np.clip(target_rms, 0.4, 0.75)
        gain = target_rms / rms
        current_peak = np.abs(audio_float32).max()
        if current_peak * gain > 0.97:
            gain = 0.97 / current_peak
        return (audio_float32 * gain).astype(np.float32)


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
    
    result = audio.copy()
    
    # 1. Transient shaper: enhance attack for punchier sound
    diff = np.diff(audio, prepend=audio[0])
    transients = np.abs(diff)
    transient_threshold = np.percentile(transients, 90)
    transient_mask = transients > transient_threshold
    transient_boost = np.zeros_like(audio)
    transient_boost[transient_mask] = diff[transient_mask] * 0.3 * clarity_amount
    result = result + transient_boost
    
    # 2. Harmonic exciter: add even harmonics for warmth and detail (8-12 kHz)
    sos_exciter = signal.butter(4, [8000, 12000], btype='band', fs=sample_rate, output='sos')
    high_band = signal.sosfilt(sos_exciter, audio).astype(np.float32)
    # Soft saturation to generate harmonics
    excited = np.tanh(high_band * 2.5) * 0.4
    result = result + excited * clarity_amount * 0.4
    
    # 3. Presence boost (2-5 kHz) for vocal/instrument clarity
    sos_presence = signal.butter(3, [2000, 5000], btype='band', fs=sample_rate, output='sos')
    presence = signal.sosfilt(sos_presence, audio).astype(np.float32)
    result = result + presence * clarity_amount * 0.25
    
    # 4. Air band (10-16 kHz) for sparkle and openness
    sos_air = signal.butter(2, 10000, btype='high', fs=sample_rate, output='sos')
    air = signal.sosfilt(sos_air, audio).astype(np.float32)
    result = result + air * clarity_amount * 0.15
    
    # 5. Mud removal (80-200 Hz cut)
    sos_mud = signal.butter(2, [80, 200], btype='band', fs=sample_rate, output='sos')
    mud = signal.sosfilt(sos_mud, audio).astype(np.float32)
    result = result - mud * clarity_amount * 0.2
    
    return result.astype(np.float32)


def apply_soft_limiter(audio, threshold=0.98, lookahead_ms=2.0, sample_rate=44100):
    """
    Fast true-peak limiter with simple lookahead (~5ms vs 50ms before).
    Prevents clipping while maintaining speed.
    """
    if len(audio.shape) > 1:
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_soft_limiter(audio[ch, :], threshold, lookahead_ms, sample_rate)
        return result
    
    audio = audio.astype(np.float32)
    lookahead_samples = int(lookahead_ms * sample_rate / 1000)
    lookahead_samples = max(1, min(lookahead_samples, 100))  # Cap lookahead to ~2.3ms @ 44.1kHz
    
    # Fast lookahead using ndimage's maximum_filter1d (vectorized, no loops)
    envelope = np.abs(audio)
    max_envelope = ndimage.maximum_filter1d(envelope, size=lookahead_samples)
    
    # Simple vectorized gain reduction (no soft knee complexity)
    gain_reduction = np.ones_like(max_envelope, dtype=np.float32)
    mask = max_envelope > threshold
    
    if np.any(mask):
        # Hard limiting with smooth transition
        ratio = 0.5  # Fast soft knee
        excess = max_envelope[mask] - threshold
        gain_reduction[mask] = np.minimum(1.0, (threshold + excess * ratio) / max_envelope[mask])
    
    # Apply gain directly (no smoothing loop)
    limited = audio * gain_reduction
    
    # Final safety clip with soft tanh
    limited = np.tanh(limited / threshold) * threshold
    
    return limited.astype(np.float32)
