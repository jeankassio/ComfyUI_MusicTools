"""
Utility functions for audio processing in ComfyUI Music Tools
"""

import math

import numpy as np
import scipy.signal as signal
from scipy.fft import fft
import torch

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    pyln = None
    HAS_PYLOUDNORM = False

try:
    from comfy.utils import ProgressBar
    HAS_PROGRESS_BAR = True
except ImportError:
    HAS_PROGRESS_BAR = False


def audio_to_numpy(audio, allow_nonfinite=False):
    """
    Convert audio from ComfyUI format to a canonical NumPy array.
    
    ComfyUI audio format:
    {
        "waveform": torch.Tensor shape (batch, channels, samples),
        "sample_rate": int
    }
    
    Args:
        audio: Audio dict from ComfyUI
        allow_nonfinite: Preserve NaN/Inf for a repair stage instead of rejecting them
    
    Returns:
        tuple: (audio_numpy: np.ndarray shape [batch, channels, samples], sample_rate: int)
    """
    if not isinstance(audio, dict):
        raise ValueError(f"Expected dict, got {type(audio)}")
    
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Audio dict must have 'waveform' and 'sample_rate' keys")
    
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    # Convert tensor to numpy
    if isinstance(waveform, torch.Tensor):
        audio_np = waveform.cpu().detach().numpy()
    elif isinstance(waveform, np.ndarray):
        audio_np = waveform
    else:
        raise ValueError(f"Unsupported waveform type: {type(waveform)}")
    
    # Canonical ComfyUI layout is [B, C, T]. Accept common unbatched forms too.
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, np.newaxis, :]
    elif audio_np.ndim == 2:
        audio_np = audio_np[np.newaxis, :, :]
    elif audio_np.ndim != 3:
        raise ValueError(
            f"Expected waveform shape [B, C, T], [C, T], or [T], got {audio_np.shape}"
        )

    if audio_np.shape[0] < 1 or audio_np.shape[1] < 1:
        raise ValueError(f"Waveform must contain at least one batch and channel, got {audio_np.shape}")
    if not allow_nonfinite and not np.isfinite(audio_np).all():
        raise ValueError("Waveform contains NaN or infinite values")

    # NumPy can otherwise share storage with a CPU float32 torch tensor. Audio
    # processors are allowed to work in-place internally, but must never mutate
    # the AUDIO value feeding another ComfyUI branch.
    return np.ascontiguousarray(audio_np).copy(), sample_rate


def numpy_to_audio_tensor(audio_np, sample_rate=44100):
    """
    Convert numpy array back to ComfyUI audio format.
    
    Args:
        audio_np: NumPy array shape [B, C, T], [C, T], or [T]
        sample_rate: sample rate in Hz (default 44100)
    
    Returns:
        dict: ComfyUI audio format {"waveform": torch.Tensor (batch, channels, samples), "sample_rate": int}
    """
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")

    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[np.newaxis, np.newaxis, :]
    elif audio_np.ndim == 2:
        audio_np = audio_np[np.newaxis, :, :]
    elif audio_np.ndim != 3:
        raise ValueError(f"Expected audio with 1, 2, or 3 dimensions, got {audio_np.shape}")
    if not np.isfinite(audio_np).all():
        raise ValueError("Processed waveform contains NaN or infinite values")

    audio_tensor = torch.from_numpy(np.ascontiguousarray(audio_np))
    return {
        "waveform": audio_tensor,
        "sample_rate": sample_rate
    }


def get_progress_bar(total, label="Processing"):
    """
    Create a progress bar for ComfyUI.
    
    Usage:
        pbar = get_progress_bar(100, "Noise Removal")
        for i in range(100):
            # do work
            pbar.update_absolute(i + 1)
    
    Args:
        total: Total number of steps
        label: Display label (informational)
    
    Returns:
        ProgressBar object or dummy object if not available
    """
    if HAS_PROGRESS_BAR:
        return ProgressBar(total)
    else:
        # Dummy progress bar
        class DummyProgressBar:
            def update_absolute(self, value, total=None, preview=None):
                pass
            def update(self, value):
                pass
        return DummyProgressBar()


def ensure_mono(audio_data):
    """Convert audio to mono while preserving an optional batch dimension."""
    audio_data = np.asarray(audio_data)
    if audio_data.ndim == 3:
        return np.mean(audio_data, axis=1, keepdims=True)
    if audio_data.ndim == 2:
        return np.mean(audio_data, axis=0, keepdims=True)
    if audio_data.ndim == 1:
        return audio_data[np.newaxis, :]
    raise ValueError(f"Unsupported audio shape: {audio_data.shape}")


def ensure_stereo(audio_data):
    """Duplicate mono audio to stereo, preserving an optional batch dimension."""
    audio_data = np.asarray(audio_data)
    if audio_data.ndim == 3:
        if audio_data.shape[1] == 1:
            return np.repeat(audio_data, 2, axis=1)
        return audio_data
    if audio_data.ndim == 2:
        if audio_data.shape[0] == 1:
            return np.repeat(audio_data, 2, axis=0)
        return audio_data
    if audio_data.ndim == 1:
        return np.stack([audio_data, audio_data], axis=0)
    raise ValueError(f"Unsupported audio shape: {audio_data.shape}")


def spectral_subtraction(audio, intensity=0.5, sample_rate=44100, pbar=None):
    """
    Remove noise using professional noise reduction.
    Uses noisereduce library for best results, falls back to simple gain reduction.
    
    Args:
        audio: Audio data numpy array shape (channels, samples)
        intensity: Noise removal intensity (0-1)
        sample_rate: Sample rate of the audio
        pbar: Optional progress bar object
    
    Returns:
        Noise-reduced audio same shape as input (channels, samples)
    """
    # Ensure audio is float32 and correct shape
    audio = audio.astype(np.float32)
    
    if pbar:
        pbar.update_absolute(15)
    
    # Audio should be (channels, samples) from audio_to_numpy
    if len(audio.shape) != 2:
        raise ValueError(f"Expected shape (channels, samples), got {audio.shape}")
    
    n_channels, n_samples = audio.shape
    
    print(f"[Noise Removal] Processing {n_channels} channels, {n_samples} samples")
    
    # Use noisereduce library if available
    if HAS_NOISEREDUCE:
        try:
            # Map intensity (0-1) to prop_decrease (0.0-1.0)
            prop_decrease = np.clip(intensity, 0.0, 1.0)
            
            result = np.zeros_like(audio)
            
            # Estimate noise profile from first 5% of audio
            noise_duration_samples = max(sample_rate // 20, 1024)  # At least 1024 samples
            noise_duration_samples = min(noise_duration_samples, n_samples // 10)
            
            print(f"[Noise Removal] Noise profile: {noise_duration_samples} samples, prop_decrease={prop_decrease:.2f}")
            
            # Process each channel
            for ch in range(n_channels):
                try:
                    channel_audio = audio[ch, :]  # Get channel as 1D array (channels, samples)
                    
                    # Validate channel audio
                    if len(channel_audio) < 2048:
                        # Too short, use input as-is
                        print(f"[Noise Removal] Channel {ch}: Too short ({len(channel_audio)} samples), skipping")
                        result[ch, :] = channel_audio
                        continue
                    
                    # Create noise sample
                    noise_sample = channel_audio[:noise_duration_samples].copy()
                    
                    if len(noise_sample) < 512:
                        print(f"[Noise Removal] Channel {ch}: Noise sample too short ({len(noise_sample)} samples), skipping")
                        result[ch, :] = channel_audio
                        continue
                    
                    print(f"[Noise Removal] Channel {ch}: Processing {len(channel_audio)} samples...")
                    
                    # Apply noise reduction
                    reduced = nr.reduce_noise(
                        y=channel_audio,
                        sr=sample_rate,
                        prop_decrease=prop_decrease,
                        y_noise=noise_sample,
                        stationary=True,
                        n_jobs=1,
                        chunk_size=600000,
                        padding=30000
                    )
                    
                    result[ch, :] = reduced
                    print(f"[Noise Removal] Channel {ch}: Success")
                    
                    if pbar:
                        pbar.update_absolute(15 + (ch + 1) * (70 // n_channels))
                
                except Exception as ch_err:
                    # If noisereduce fails, use simple attenuation
                    print(f"[Noise Removal] Channel {ch} failed ({type(ch_err).__name__}: {str(ch_err)[:100]}), using simple attenuation")
                    # Simple noise reduction: reduce overall amplitude slightly
                    result[ch, :] = audio[ch, :] * (1.0 - intensity * 0.1)
            
            if pbar:
                pbar.update_absolute(85)
            
            print(f"[Noise Removal] Complete")
            return result
            
        except Exception as e:
            print(f"[Noise Removal] Library error ({type(e).__name__}: {str(e)[:100]}), using simple attenuation")
            if pbar:
                pbar.update_absolute(20)
            # Simple fallback: slight attenuation based on intensity
            return audio * (1.0 - intensity * 0.1)
    
    # If noisereduce not available, use simple attenuation
    print(f"[Noise Removal] noisereduce not available, using simple attenuation")
    if pbar:
        pbar.update_absolute(85)
    return audio * (1.0 - intensity * 0.1)


def upscale_audio(audio, target_sr=48000, sample_rate=44100):
    """
    Upscale audio to a higher sample rate using librosa or scipy.
    Uses librosa for better quality resampling when available.
    
    Args:
        audio: Audio data shape (channels, samples)
        target_sr: Target sample rate
        sample_rate: Original sample rate
    
    Returns:
        Upscaled audio same channels, new length
    """
    sample_rate = int(sample_rate)
    target_sr = int(target_sr)
    if sample_rate <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim < 1 or audio.ndim > 3:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")
    if sample_rate == target_sr or audio.shape[-1] == 0:
        return audio.copy()

    # Polyphase resampling is band-limited, deterministic, and handles every
    # leading batch/channel dimension along the final time axis.
    divisor = math.gcd(sample_rate, target_sr)
    up = target_sr // divisor
    down = sample_rate // divisor
    result = signal.resample_poly(audio, up, down, axis=-1)

    # resample_poly uses ceil for the output length. Enforce that documented
    # duration exactly so all batches and downstream mixers agree.
    expected_length = int(round(audio.shape[-1] * target_sr / sample_rate))
    if result.shape[-1] > expected_length:
        result = result[..., :expected_length]
    elif result.shape[-1] < expected_length:
        pad_width = [(0, 0)] * result.ndim
        pad_width[-1] = (0, expected_length - result.shape[-1])
        result = np.pad(result, pad_width)
    return result.astype(np.float32, copy=False)


def restore_frequency(audio, original_sr=44100, upscaled_sr=48000, current_sr=None):
    """
    Restore audio to original frequency using interpolation.
    
    Args:
        audio: Upscaled audio data shape (channels, samples)
        original_sr: Original sample rate
        upscaled_sr: Upscaled sample rate
        current_sr: If specified, treat audio as having this sample rate
    
    Returns:
        Audio restored to original sample rate shape (channels, samples)
    """
    if current_sr is None:
        current_sr = upscaled_sr
    return upscale_audio(audio, target_sr=original_sr, sample_rate=current_sr)


def enhance_stereo(audio, intensity=0.5):
    """
    Enhance stereo separation.
    
    Args:
        audio: Stereo audio data shape (channels, samples)
        intensity: Enhancement intensity (0-1)
    
    Returns:
        Enhanced stereo audio shape (channels, samples)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 3:
        return np.stack([enhance_stereo(batch, intensity) for batch in audio], axis=0)
    if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[-1] == 0:
        return audio.copy()
    
    # Calculate mid and side signals
    mid = (audio[0, :] + audio[1, :]) / 2
    side = (audio[0, :] - audio[1, :]) / 2
    
    # Enhance side channel
    side_enhanced = side * (1.0 + float(np.clip(intensity, 0.0, 2.0)))
    
    # Convert back to stereo
    left = mid + side_enhanced
    right = mid - side_enhanced
    
    # Normalize to prevent clipping
    max_val = max(np.abs(left).max(), np.abs(right).max())
    if max_val > 1.0:
        left = left / max_val
        right = right / max_val
    
    return np.stack([left, right], axis=0).astype(np.float32)


def calculate_lufs(audio, sample_rate=44100):
    """
    Calculate integrated loudness using ITU-R BS.1770 when pyloudnorm is present.
    
    Args:
        audio: Audio data shape (channels, samples)
        sample_rate: Sample rate
    
    Returns:
        LUFS value
    """
    audio = np.asarray(audio, dtype=np.float32)
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if audio.ndim == 3:
        values = [calculate_lufs(batch, sample_rate) for batch in audio]
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("-inf")
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.ndim != 2:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")
    if audio.shape[-1] == 0 or not np.any(audio):
        return float("-inf")

    # pyloudnorm expects [samples, channels] (or a mono vector). It supports
    # normal program material and performs K-weighting plus absolute/relative
    # gating. Very short clips fall through to the deterministic RMS estimate.
    if HAS_PYLOUDNORM and pyln is not None:
        try:
            meter_input = audio[0] if audio.shape[0] == 1 else audio.T
            if audio.shape[0] > 5:
                meter_input = np.mean(audio, axis=0)
            meter = pyln.Meter(sample_rate)
            loudness = float(meter.integrated_loudness(meter_input))
            if np.isfinite(loudness):
                return loudness
        except (ValueError, RuntimeError, FloatingPointError):
            pass

    # BS.1770 uses -0.691 LKFS for a full-scale, K-weighted reference signal.
    # This unweighted fallback is intended only for clips too short for gating.
    mean_power = float(np.mean(np.sum(np.square(audio, dtype=np.float64), axis=0)))
    if mean_power <= np.finfo(np.float64).tiny:
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(mean_power))


def normalize_to_lufs(audio, target_lufs=-14, sample_rate=44100, peak_ceiling_db=-1.0):
    """
    Normalize audio to target LUFS.
    
    Args:
        audio: Audio data
        target_lufs: Target LUFS value
        sample_rate: Sample rate
    
    Returns:
        LUFS-normalized audio
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 3:
        return np.stack(
            [normalize_to_lufs(batch, target_lufs, sample_rate, peak_ceiling_db) for batch in audio],
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

    current_lufs = calculate_lufs(work, sample_rate)
    if not np.isfinite(current_lufs):
        return audio.copy()

    gain_linear = 10.0 ** ((float(target_lufs) - current_lufs) / 20.0)
    normalized = work * gain_linear

    # LUFS gain can request impossible loudness for highly dynamic material.
    # Keep a safe sample-peak ceiling; the standalone true-peak limiter can be
    # placed after this node when strict dBTP delivery is required.
    ceiling = 10.0 ** (float(peak_ceiling_db) / 20.0)
    peak = float(np.max(np.abs(normalized)))
    if peak > ceiling > 0.0:
        normalized *= ceiling / peak

    normalized = normalized.astype(np.float32, copy=False)
    return normalized[0] if remove_channel else normalized


def apply_eq(audio, frequencies, gains, sample_rate=44100):
    """
    Apply parametric EQ to audio.
    
    Args:
        audio: Audio data shape (channels, samples)
        frequencies: List of center frequencies
        gains: List of gain values in dB
        sample_rate: Sample rate
    
    Returns:
        Equalized audio shape (channels, samples)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        return np.stack(
            [apply_eq(channel, frequencies, gains, sample_rate) for channel in audio],
            axis=0,
        )
    if audio.size == 0:
        return audio.copy()
    
    # Mono processing
    result = audio.copy().astype(np.float32)
    
    for freq, gain_db in zip(frequencies, gains):
        if gain_db == 0:
            continue
        
        nyquist = sample_rate / 2.0
        freq = float(np.clip(freq, 1.0, nyquist * 0.98))
        gain_db = float(np.clip(gain_db, -24.0, 24.0))
        A = 10.0 ** (gain_db / 40.0)
        Q = 1.0
        
        # Peaking filter coefficients
        w0 = 2 * np.pi * freq / sample_rate
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = sin_w0 / (2 * Q)
        
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
        
        # Normalize coefficients
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        
        # A causal biquad also works on very short clips, unlike filtfilt.
        result = signal.lfilter(b, a, result)
    
    return result.astype(np.float32)


def apply_reverb(audio, decay=0.5, sample_rate=44100):
    """
    Apply simple reverb effect using delay and feedback.
    
    Args:
        audio: Audio data shape (channels, samples)
        decay: Reverb decay factor (0-1)
        sample_rate: Sample rate
    
    Returns:
        Reverbed audio shape (channels, samples)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        # Process stereo
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_reverb(audio[ch, :], decay, sample_rate)
        return result
    
    if audio.size == 0 or decay <= 0.0:
        return audio.copy()

    # Mono processing - simple multi-tap reverberator
    delay_times = [0.029, 0.031, 0.037, 0.041]  # In seconds
    delay_samples = [int(dt * sample_rate) for dt in delay_times]
    
    output = np.zeros(len(audio) + max(delay_samples), dtype=np.float32)
    
    for delay in delay_samples:
        delayed = np.zeros(len(audio) + delay, dtype=np.float32)
        delayed[delay:] = audio.astype(np.float32)
        output[:len(delayed)] += delayed * decay
    
    # Mix with original
    result = np.zeros_like(audio)
    result[:] = audio.astype(np.float32) + output[:len(audio)] * 0.5
    
    # Normalize
    max_val = np.abs(result).max()
    if max_val > 1.0:
        result = result / max_val
    
    return result


def apply_compression(
    audio,
    threshold=0.5,
    ratio=4.0,
    sample_rate=44100,
    attack_ms=5.0,
    release_ms=80.0,
):
    """
    Apply dynamic range compression.
    
    Args:
        audio: Audio data shape (channels, samples)
        threshold: Compression threshold (0-1)
        ratio: Compression ratio
        sample_rate: Sample rate
        attack_ms: Detector attack time in milliseconds
        release_ms: Gain recovery time in milliseconds
    
    Returns:
        Compressed audio shape (channels, samples)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 3:
        return np.stack(
            [
                apply_compression(
                    batch, threshold, ratio, sample_rate, attack_ms, release_ms
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
    if work.shape[-1] == 0 or ratio <= 1.0:
        return audio.copy()

    # A linked peak detector preserves the stereo image. Gain is computed in
    # dB with causal attack/release envelopes, avoiding the waveshaping
    # distortion caused by compressing individual samples.
    threshold = float(np.clip(threshold, 1e-4, 1.0))
    ratio = max(float(ratio), 1.0)
    detector = np.max(np.abs(work), axis=0)
    attack_samples = max(1.0, float(attack_ms) * sample_rate / 1000.0)
    attack_alpha = float(np.exp(-1.0 / attack_samples))
    envelope, _ = signal.lfilter(
        [1.0 - attack_alpha],
        [1.0, -attack_alpha],
        detector,
        zi=[attack_alpha * detector[0]],
    )
    level_db = 20.0 * np.log10(np.maximum(envelope, 1e-12))
    threshold_db = 20.0 * np.log10(threshold)
    gain_db = -np.maximum(level_db - threshold_db, 0.0) * (1.0 - 1.0 / ratio)
    desired_gain = np.power(10.0, gain_db / 20.0).astype(np.float32)

    release_samples = max(1.0, float(release_ms) * sample_rate / 1000.0)
    alpha = float(np.exp(-1.0 / release_samples))
    smoothed, _ = signal.lfilter(
        [1.0 - alpha],
        [1.0, -alpha],
        desired_gain,
        zi=[alpha * desired_gain[0]],
    )
    gain = np.minimum(smoothed, desired_gain).astype(np.float32)
    result = (work * gain[np.newaxis, :]).astype(np.float32)
    return result[0] if remove_channel else result


def apply_gain(audio, gain_db=0):
    """
    Apply gain to audio.
    
    Args:
        audio: Audio data
        gain_db: Gain in dB
    
    Returns:
        Audio with applied gain
    """
    gain_linear = 10 ** (gain_db / 20)
    return (np.asarray(audio, dtype=np.float32) * gain_linear).astype(np.float32)


def mix_audio(*audio_samples):
    """
    Mix multiple audio samples together.
    
    Args:
        *audio_samples: Variable number of audio samples shape (channels, samples)
    
    Returns:
        Mixed audio shape (channels, samples)
    """
    if not audio_samples:
        raise ValueError("At least one audio sample is required")
    
    original_ndim = max(np.asarray(item).ndim for item in audio_samples)

    def canonical(item):
        item = np.asarray(item, dtype=np.float32)
        if item.ndim == 1:
            return item[np.newaxis, np.newaxis, :]
        if item.ndim == 2:
            return item[np.newaxis, :, :]
        if item.ndim == 3:
            return item
        raise ValueError(f"Unsupported audio shape: {item.shape}")

    canonical_audio = [canonical(item) for item in audio_samples]
    batch_count = max(item.shape[0] for item in canonical_audio)
    channel_count = max(item.shape[1] for item in canonical_audio)
    output_length = max(item.shape[-1] for item in canonical_audio)
    result = np.zeros((batch_count, channel_count, output_length), dtype=np.float32)

    for item in canonical_audio:
        if item.shape[0] == 1 and batch_count > 1:
            item = np.repeat(item, batch_count, axis=0)
        elif item.shape[0] != batch_count:
            raise ValueError("Audio batch sizes must match or be broadcastable from one")
        if item.shape[1] == 1 and channel_count > 1:
            item = np.repeat(item, channel_count, axis=1)
        elif item.shape[1] != channel_count:
            raise ValueError("Audio channel counts must match or be mono")
        result[..., :item.shape[-1]] += item

    if result.size:
        # Batch items are independent ComfyUI values. A hot item must not turn
        # down another item merely because they share the same tensor.
        peaks = np.max(np.abs(result), axis=(1, 2), keepdims=True)
        result /= np.maximum(peaks, 1.0)

    if original_ndim == 1:
        return result[0, 0]
    if original_ndim == 2:
        return result[0]
    return result


def trim_audio(audio, start_time=0, end_time=None, sample_rate=44100):
    """
    Trim audio to specified time range.
    
    Args:
        audio: Audio data shape (channels, samples)
        start_time: Start time in seconds
        end_time: End time in seconds (None = end of file)
        sample_rate: Sample rate
    
    Returns:
        Trimmed audio shape (channels, samples)
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim < 1 or audio.ndim > 3:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")
    total_samples = audio.shape[-1]
    start_sample = int(round(float(start_time) * sample_rate))
    end_sample = total_samples if end_time is None else int(round(float(end_time) * sample_rate))
    start_sample = max(0, min(start_sample, total_samples))
    end_sample = max(0, min(end_sample, total_samples))
    if start_sample >= end_sample:
        raise ValueError("Start time must be before end time and inside the audio duration")
    return audio[..., start_sample:end_sample].astype(np.float32, copy=True)


def separate_stems(audio, separation_type="vocals", sample_rate=44100):
    """
    Separate audio into stems using frequency-based and harmonic-percussive analysis.
    Simplified implementation without deep learning.
    
    Args:
        audio: Audio data shape (channels, samples)
        separation_type: Type of separation
            - "vocals": Extract vocals/lead
            - "drums": Extract drums/percussion
            - "bass": Extract bass
            - "music": Extract melody/instruments
            - "others": Extract residual audio
        sample_rate: Sample rate
    
    Returns:
        Separated audio stem shape (channels, samples)
    """
    if len(audio.shape) > 1:
        # Process stereo
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = separate_stems(audio[ch, :], separation_type, sample_rate)
        return result
    
    # Mono processing using STFT
    frame_length = 2048
    hop_length = frame_length // 4
    
    # Compute STFT
    n_frames = (len(audio) - frame_length) // hop_length + 1
    stft_matrix = np.zeros((frame_length // 2 + 1, n_frames), dtype=np.complex64)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        
        if end > len(audio):
            break
        
        frame = audio[start:end].astype(np.float32) * signal.windows.hann(frame_length)
        stft_matrix[:, i] = fft(frame)[:frame_length // 2 + 1]
    
    # Frequency-based separation
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    # Define frequency ranges (in bins)
    freq_resolution = sample_rate / frame_length
    
    if separation_type == "vocals":
        # Vocals: 200 Hz - 8 kHz (presence peak)
        min_freq_bin = int(200 / freq_resolution)
        max_freq_bin = int(8000 / freq_resolution)
    elif separation_type == "drums":
        # Drums: 0 - 6 kHz with emphasis on peaks
        min_freq_bin = 0
        max_freq_bin = int(6000 / freq_resolution)
    elif separation_type == "bass":
        # Bass: 20 Hz - 250 Hz
        min_freq_bin = int(20 / freq_resolution)
        max_freq_bin = int(250 / freq_resolution)
    else:  # "music" / melody
        # Everything else
        min_freq_bin = 0
        max_freq_bin = magnitude.shape[0] - 1
    
    # Apply frequency mask
    mask = np.zeros_like(magnitude)
    mask[min_freq_bin:max_freq_bin] = 1.0
    
    # Apply percussion envelope for drums
    if separation_type == "drums":
        # Enhance attack transients
        onset_env = np.abs(np.diff(magnitude, axis=1, prepend=0))
        onset_mask = onset_env > np.percentile(onset_env, 70)
        mask = mask * 0.5 + onset_mask.astype(float) * 0.5
    
    # Apply harmonic-percussive separation using median filtering
    if separation_type in ["vocals", "music"]:
        # Harmonic separation using median filter
        harmonic = signal.medfilt(magnitude, kernel_size=(11, 1))
        mask = harmonic / (magnitude + 1e-10)
        mask = np.clip(mask, 0, 1)
    
    # Apply mask
    separated_magnitude = magnitude * mask
    
    # Reconstruct STFT
    separated_stft = separated_magnitude * np.exp(1j * phase)
    
    # Inverse STFT
    result = np.zeros(len(audio), dtype=np.float32)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        
        if end > len(audio):
            break
        
        # Pad STFT to original frame length
        padded_stft = np.zeros(frame_length, dtype=np.complex64)
        padded_stft[:frame_length // 2 + 1] = separated_stft[:, i]
        padded_stft[frame_length // 2 + 1:] = np.conj(separated_stft[frame_length // 2 - 1:0:-1, i])
        
        frame_result = np.real(np.fft.ifft(padded_stft))
        frame_result = frame_result * signal.windows.hann(frame_length)
        
        result[start:end] += frame_result
    
    # Normalize only if not "others" - others will be normalized by recombine_stems
    if separation_type != "others":
        result = result / np.max(np.abs(result) + 1e-10)
    
    return result.astype(np.float32)


def separate_all_stems(audio, sample_rate=44100):
    """
    Separate audio into all main stems plus 'others'.
    The 'others' stem captures audio not in vocals, drums, bass, or music.
    
    Args:
        audio: Audio data shape (channels, samples)
        sample_rate: Sample rate
    
    Returns:
        Dictionary with stems: vocals, drums, bass, music, others
        All stems normalized individually to peak ~1.0
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 3:
        separated_batches = [separate_all_stems(batch, sample_rate) for batch in audio]
        return {
            name: np.stack([batch[name] for batch in separated_batches], axis=0)
            for name in ("vocals", "drums", "bass", "music", "others")
        }
    if audio.ndim not in (1, 2):
        raise ValueError(f"Unsupported audio shape: {audio.shape}")

    # Very short clips do not contain enough context for STFT/median masks.
    # Preserve them losslessly in the music stem instead of failing with a
    # negative frame count.
    if audio.shape[-1] < 2048:
        zeros = np.zeros_like(audio, dtype=np.float32)
        return {
            "vocals": zeros.copy(),
            "drums": zeros.copy(),
            "bass": zeros.copy(),
            "music": audio.copy(),
            "others": zeros.copy(),
        }

    # Extract each stem separately using STFT-based approach
    # WITHOUT normalizing individual stems (keep linear scale)
    # Then calculate others as residual
    # Finally normalize all stems for output
    
    # If mono, add channel dimension for consistent processing
    if len(audio.shape) < 2:
        audio = audio[np.newaxis, :]
        remove_channel = True
    else:
        remove_channel = False
    
    # Process all 4 stems first (unnormalized, linear scale)
    stems_unnormalized = {}
    
    for stem_type in ["vocals", "drums", "bass", "music"]:
        # Extract this stem type using separate_stems (which normalizes it)
        # But we'll recalculate without normalization
        stem_data = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            audio_ch = audio[ch, :]
            
            # Process this channel for this stem type
            frame_length = 2048
            hop_length = frame_length // 4
            
            # Compute STFT
            n_frames = (len(audio_ch) - frame_length) // hop_length + 1
            stft_matrix = np.zeros((frame_length // 2 + 1, n_frames), dtype=np.complex64)
            
            for i in range(n_frames):
                start = i * hop_length
                end = start + frame_length
                if end > len(audio_ch):
                    break
                frame = audio_ch[start:end].astype(np.float32) * signal.windows.hann(frame_length)
                stft_matrix[:, i] = fft(frame)[:frame_length // 2 + 1]
            
            magnitude = np.abs(stft_matrix)
            phase = np.angle(stft_matrix)
            freq_resolution = sample_rate / frame_length
            
            # Define mask for this stem type
            if stem_type == "vocals":
                min_freq_bin = int(200 / freq_resolution)
                max_freq_bin = int(8000 / freq_resolution)
            elif stem_type == "drums":
                min_freq_bin = 0
                max_freq_bin = int(6000 / freq_resolution)
            elif stem_type == "bass":
                min_freq_bin = int(20 / freq_resolution)
                max_freq_bin = int(250 / freq_resolution)
            else:  # "music"
                min_freq_bin = 0
                max_freq_bin = magnitude.shape[0] - 1
            
            # Create frequency mask
            mask = np.zeros_like(magnitude)
            mask[min_freq_bin:max_freq_bin] = 1.0
            
            # Special processing
            if stem_type == "drums":
                onset_env = np.abs(np.diff(magnitude, axis=1, prepend=0))
                onset_mask = onset_env > np.percentile(onset_env, 70)
                mask = mask * 0.5 + onset_mask.astype(float) * 0.5
            elif stem_type in ["vocals", "music"]:
                harmonic = signal.medfilt(magnitude, kernel_size=(11, 1))
                harmonic_mask = np.clip(harmonic / (magnitude + 1e-10), 0, 1)
                # Keep the frequency range selected above. The old assignment
                # replaced it, making the vocals and music outputs identical.
                mask *= harmonic_mask
            
            # Apply mask and inverse STFT
            separated_magnitude = magnitude * mask
            separated_stft = separated_magnitude * np.exp(1j * phase)
            
            result_ch = np.zeros(len(audio_ch), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop_length
                end = start + frame_length
                if end > len(audio_ch):
                    break
                padded_stft = np.zeros(frame_length, dtype=np.complex64)
                padded_stft[:frame_length // 2 + 1] = separated_stft[:, i]
                padded_stft[frame_length // 2 + 1:] = np.conj(separated_stft[frame_length // 2 - 1:0:-1, i])
                frame_result = np.real(np.fft.ifft(padded_stft))
                frame_result = frame_result * signal.windows.hann(frame_length)
                result_ch[start:end] += frame_result
            
            stem_data[ch, :] = result_ch
        
        stems_unnormalized[stem_type] = stem_data
    
    # Calculate 'others' as residual in linear scale
    others = audio.copy()
    for stem_type in ["vocals", "drums", "bass", "music"]:
        others = others - stems_unnormalized[stem_type]
    
    # Keep every stem on the original linear scale. Independent peak
    # normalization destroyed their balance and made unity recombination
    # impossible. The residual guarantees that all five stems sum to input.
    stems = {
        "vocals": stems_unnormalized["vocals"].astype(np.float32),
        "drums": stems_unnormalized["drums"].astype(np.float32),
        "bass": stems_unnormalized["bass"].astype(np.float32),
        "music": stems_unnormalized["music"].astype(np.float32),
        "others": others.astype(np.float32),
    }
    
    # Remove channel dimension if it was added
    if remove_channel:
        for key in stems:
            stems[key] = stems[key][0, :]
    
    return stems


def recombine_stems(stems_dict, weights=None):
    """
    Recombine separated stems back into a full mix.
    Uses proper mixing with peak normalization to preserve loudness and dynamics.
    
    Args:
        stems_dict: Dictionary with stem names and audio data shape (channels, samples)
        weights: Optional dictionary with stem weights (0-2, default 1.0)
    
    Returns:
        Recombined audio shape (channels, samples)
    """
    if not stems_dict:
        raise ValueError("At least one stem is required")
    if weights is None:
        weights = {stem: 1.0 for stem in stems_dict.keys()}

    original_ndim = max(np.asarray(item).ndim for item in stems_dict.values())

    def canonical(item):
        item = np.asarray(item, dtype=np.float32)
        if item.ndim == 1:
            return item[np.newaxis, np.newaxis, :]
        if item.ndim == 2:
            return item[np.newaxis, :, :]
        if item.ndim == 3:
            return item
        raise ValueError(f"Unsupported stem shape: {item.shape}")

    canonical_stems = {name: canonical(item) for name, item in stems_dict.items()}
    batch_count = max(item.shape[0] for item in canonical_stems.values())
    channel_count = max(item.shape[1] for item in canonical_stems.values())
    output_length = max(item.shape[-1] for item in canonical_stems.values())
    result = np.zeros((batch_count, channel_count, output_length), dtype=np.float32)

    for stem_name, item in canonical_stems.items():
        if item.shape[0] == 1 and batch_count > 1:
            item = np.repeat(item, batch_count, axis=0)
        elif item.shape[0] != batch_count:
            raise ValueError("Stem batch sizes must match or be broadcastable from one")
        if item.shape[1] == 1 and channel_count > 1:
            item = np.repeat(item, channel_count, axis=1)
        elif item.shape[1] != channel_count:
            raise ValueError("Stem channel counts must match or be mono")
        result[..., :item.shape[-1]] += item * float(weights.get(stem_name, 1.0))

    if result.size:
        peaks = np.max(np.abs(result), axis=(1, 2), keepdims=True)
        result *= np.minimum(1.0, 0.98 / np.maximum(peaks, 1e-12))

    if original_ndim == 1:
        return result[0, 0]
    if original_ndim == 2:
        return result[0]
    return result.astype(np.float32, copy=False)
