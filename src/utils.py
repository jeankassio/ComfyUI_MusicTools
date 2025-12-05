"""
Utility functions for audio processing in ComfyUI Music Tools
"""

import numpy as np
import scipy.signal as signal
from scipy.fft import fft, ifft
import warnings
import torch

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from comfy.utils import ProgressBar
    HAS_PROGRESS_BAR = True
except ImportError:
    HAS_PROGRESS_BAR = False


def audio_to_numpy(audio):
    """
    Convert audio from ComfyUI format to numpy array.
    
    ComfyUI audio format:
    {
        "waveform": torch.Tensor shape (channels, samples),
        "sample_rate": int
    }
    
    Note: After .squeeze(0) if batch dimension exists
    
    Args:
        audio: Audio dict from ComfyUI
    
    Returns:
        tuple: (audio_numpy: np.ndarray shape [channels, samples], sample_rate: int)
    """
    if not isinstance(audio, dict):
        raise ValueError(f"Expected dict, got {type(audio)}")
    
    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Audio dict must have 'waveform' and 'sample_rate' keys")
    
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    
    # Convert tensor to numpy
    if isinstance(waveform, torch.Tensor):
        audio_np = waveform.cpu().detach().numpy()
    elif isinstance(waveform, np.ndarray):
        audio_np = waveform
    else:
        raise ValueError(f"Unsupported waveform type: {type(waveform)}")
    
    # Ensure float32
    audio_np = audio_np.astype(np.float32)
    
    print(f"[audio_to_numpy] Raw shape: {audio_np.shape}")
    
    # Audio should be (channels, samples)
    # Sometimes may come with batch dimension as first dim
    if len(audio_np.shape) == 3:
        # Shape: (batch, channels, samples) - remove batch
        audio_np = audio_np[0]
        print(f"[audio_to_numpy] Removed batch dimension: {audio_np.shape}")
    elif len(audio_np.shape) == 1:
        # Shape: (samples,) - add channel dimension
        audio_np = audio_np[np.newaxis, :]
        print(f"[audio_to_numpy] Added channel dimension: {audio_np.shape}")
    
    # Final validation: should be (channels, samples)
    if len(audio_np.shape) != 2:
        raise ValueError(f"Expected shape (channels, samples), got {audio_np.shape}")
    
    n_channels, n_samples = audio_np.shape
    print(f"[audio_to_numpy] SUCCESS: {n_channels} channels, {n_samples} samples")
    
    return audio_np, sample_rate


def numpy_to_audio_tensor(audio_np, sample_rate=44100):
    """
    Convert numpy array back to ComfyUI audio format.
    
    Args:
        audio_np: numpy array shape [channels, samples]
        sample_rate: sample rate in Hz (default 44100)
    
    Returns:
        dict: ComfyUI audio format {"waveform": torch.Tensor (batch, channels, samples), "sample_rate": int}
    """
    # Ensure float32
    audio_np = audio_np.astype(np.float32)
    
    # Handle shape - audio_np should be (channels, samples)
    if len(audio_np.shape) == 1:
        # (samples,) -> (1, samples) - make it mono
        audio_np = audio_np[np.newaxis, :]
    
    print(f"[numpy_to_audio_tensor] Input shape: {audio_np.shape}")
    
    # Convert to tensor (channels, samples)
    audio_tensor = torch.from_numpy(audio_np)
    
    # Add batch dimension back: (channels, samples) -> (batch, channels, samples)
    audio_tensor = audio_tensor.unsqueeze(0)
    
    print(f"[numpy_to_audio_tensor] Output shape: {audio_tensor.shape}")
    
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
    """Convert audio to mono if it's stereo. Audio shape: (channels, samples) or (samples,)"""
    if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
        return np.mean(audio_data, axis=0)
    return audio_data.flatten()


def ensure_stereo(audio_data):
    """Convert audio to stereo if it's mono. Returns shape (channels, samples)"""
    if len(audio_data.shape) == 1 or audio_data.shape[0] == 1:
        mono = audio_data.flatten()
        return np.stack([mono, mono], axis=0)
    return audio_data


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
    if sample_rate == target_sr:
        return audio  # No upscaling needed
    
    audio = audio.astype(np.float32)
    n_channels, n_samples = audio.shape
    
    # Calculate new length
    new_length = int(n_samples * target_sr / sample_rate)
    result = np.zeros((n_channels, new_length), dtype=np.float32)
    
    # Use librosa if available for better quality
    if HAS_LIBROSA:
        try:
            for ch in range(n_channels):
                result[ch, :] = librosa.resample(audio[ch, :], orig_sr=sample_rate, target_sr=target_sr)
            return result
        except Exception as e:
            print(f"Warning: librosa resample failed ({e}), falling back to scipy")
    
    # Fallback: use scipy for resampling
    try:
        from scipy import signal as sp_signal  # type: ignore
        for ch in range(n_channels):
            result[ch, :] = sp_signal.resample(audio[ch, :], new_length)
        return result
    except Exception as e:
        print(f"Warning: scipy resample failed ({e}), using linear interpolation")
    
    # Fallback: linear interpolation
    indices_old = np.arange(n_samples)
    indices_new = np.linspace(0, n_samples - 1, new_length)
    
    for ch in range(n_channels):
        result[ch, :] = np.interp(indices_new, indices_old, audio[ch, :])
    
    return result


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
    
    if len(audio.shape) > 1:
        # Process stereo - shape is (channels, samples)
        old_length = audio.shape[1]
        new_length = int(old_length * original_sr / current_sr)
        result = np.zeros((audio.shape[0], new_length), dtype=np.float32)
        for ch in range(audio.shape[0]):
            result[ch, :] = restore_frequency(audio[ch, :], original_sr, upscaled_sr, current_sr)
        return result
    
    # Mono processing
    old_length = len(audio)
    new_length = int(old_length * original_sr / current_sr)
    indices_old = np.arange(old_length)
    indices_new = np.linspace(0, old_length - 1, new_length)
    
    restored = np.interp(indices_new, indices_old, audio.astype(np.float32))
    return restored.astype(np.float32)


def enhance_stereo(audio, intensity=0.5):
    """
    Enhance stereo separation.
    
    Args:
        audio: Stereo audio data shape (channels, samples)
        intensity: Enhancement intensity (0-1)
    
    Returns:
        Enhanced stereo audio shape (channels, samples)
    """
    if len(audio.shape) == 1:
        return audio  # Not stereo
    
    if audio.shape[0] != 2:
        return audio  # Not stereo
    
    # Calculate mid and side signals
    mid = (audio[0, :] + audio[1, :]) / 2
    side = (audio[0, :] - audio[1, :]) / 2
    
    # Enhance side channel
    side_enhanced = side * (1 + intensity)
    
    # Convert back to stereo
    left = mid + side_enhanced
    right = mid - side_enhanced
    
    # Normalize to prevent clipping
    max_val = max(np.abs(left).max(), np.abs(right).max())
    if max_val > 1.0:
        left = left / max_val
        right = right / max_val
    
    return np.stack([left, right], axis=0)


def calculate_lufs(audio, sample_rate=44100):
    """
    Calculate LUFS (Loudness Units relative to Full Scale) of audio.
    Simplified implementation using RMS and frequency weighting.
    
    Args:
        audio: Audio data shape (channels, samples)
        sample_rate: Sample rate
    
    Returns:
        LUFS value
    """
    if len(audio.shape) > 1:
        # Mix stereo to mono for LUFS calculation
        audio = np.mean(audio, axis=0)
    
    # Simple LUFS approximation using RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Reference level (-23 LUFS = 1.0 RMS)
    if rms > 0:
        lufs = -23 + 20 * np.log10(rms)
    else:
        lufs = -np.inf
    
    return lufs


def normalize_to_lufs(audio, target_lufs=-14, sample_rate=44100):
    """
    Normalize audio to target LUFS.
    
    Args:
        audio: Audio data
        target_lufs: Target LUFS value
        sample_rate: Sample rate
    
    Returns:
        LUFS-normalized audio
    """
    current_lufs = calculate_lufs(audio, sample_rate)
    
    if current_lufs == -np.inf or np.isnan(current_lufs):
        return audio
    
    # Calculate gain needed
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain and prevent clipping
    normalized = audio * gain_linear
    max_val = np.abs(normalized).max()
    
    if max_val > 1.0:
        normalized = normalized / max_val
    
    return normalized.astype(np.float32)


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
    if len(audio.shape) > 1:
        # Process stereo
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_eq(audio[ch, :], frequencies, gains, sample_rate)
        return result
    
    # Mono processing
    result = audio.copy().astype(np.float32)
    
    for freq, gain_db in zip(frequencies, gains):
        if gain_db == 0:
            continue
        
        # Design IIR filter
        gain_linear = 10 ** (gain_db / 20)
        Q = 1.0
        
        # Peaking filter coefficients
        w0 = 2 * np.pi * freq / sample_rate
        sin_w0 = np.sin(w0)
        cos_w0 = np.cos(w0)
        alpha = sin_w0 / (2 * Q)
        
        b0 = 1 + alpha * gain_linear
        b1 = -2 * cos_w0
        b2 = 1 - alpha * gain_linear
        a0 = 1 + alpha / gain_linear
        a1 = -2 * cos_w0
        a2 = 1 - alpha / gain_linear
        
        # Normalize coefficients
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1.0, a1/a0, a2/a0])
        
        # Apply filter
        result = signal.filtfilt(b, a, result)
    
    return result


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
    if len(audio.shape) > 1:
        # Process stereo
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_reverb(audio[ch, :], decay, sample_rate)
        return result
    
    # Mono processing - simple Schroeder reverberator
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


def apply_compression(audio, threshold=0.5, ratio=4.0, sample_rate=44100):
    """
    Apply dynamic range compression.
    
    Args:
        audio: Audio data shape (channels, samples)
        threshold: Compression threshold (0-1)
        ratio: Compression ratio
        sample_rate: Sample rate
    
    Returns:
        Compressed audio shape (channels, samples)
    """
    if len(audio.shape) > 1:
        # Process stereo
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            result[ch, :] = apply_compression(audio[ch, :], threshold, ratio, sample_rate)
        return result
    
    # Mono processing
    result = audio.copy().astype(np.float32)
    
    # Apply compression
    mask = np.abs(result) > threshold
    result[mask] = np.sign(result[mask]) * (threshold + (np.abs(result[mask]) - threshold) / ratio)
    
    return result


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
    result = audio * gain_linear
    
    # Prevent clipping
    max_val = np.abs(result).max()
    if max_val > 1.0:
        result = result / max_val
    
    return result.astype(np.float32)


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
    
    # Get common length (using shape[1] which is samples for (channels, samples))
    min_length = min(audio.shape[1] if len(audio.shape) > 1 else len(audio) for audio in audio_samples)
    n_channels = audio_samples[0].shape[0] if len(audio_samples[0].shape) > 1 else 1
    
    # Mix
    result = np.zeros((n_channels, min_length), dtype=np.float32)
    
    for audio in audio_samples:
        if len(audio.shape) > 1:
            result[:, :] += audio[:, :min_length]
        else:
            if len(result.shape) > 1:
                result[0, :] += audio[:min_length]
            else:
                result[:] += audio[:min_length]
    
    # Normalize
    max_val = np.abs(result).max()
    if max_val > 1.0:
        result = result / max_val
    
    return result.astype(np.float32)


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
    start_sample = int(start_time * sample_rate)
    
    if end_time is None:
        end_sample = audio.shape[1] if len(audio.shape) > 1 else len(audio)
    else:
        end_sample = int(end_time * sample_rate)
    
    if len(audio.shape) > 1:
        return audio[:, start_sample:end_sample].astype(np.float32)
    else:
        return audio[start_sample:end_sample].astype(np.float32)


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
                mask = harmonic / (magnitude + 1e-10)
                mask = np.clip(mask, 0, 1)
            
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
    
    # Now normalize all stems including others
    def normalize_stem(stem):
        """Normalize stem to peak ~1.0"""
        max_val = np.abs(stem).max()
        if max_val > 1e-10:
            return stem / max_val
        return stem.astype(np.float32)
    
    stems = {
        "vocals": normalize_stem(stems_unnormalized["vocals"]).astype(np.float32),
        "drums": normalize_stem(stems_unnormalized["drums"]).astype(np.float32),
        "bass": normalize_stem(stems_unnormalized["bass"]).astype(np.float32),
        "music": normalize_stem(stems_unnormalized["music"]).astype(np.float32),
        "others": normalize_stem(others).astype(np.float32),
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
    if weights is None:
        weights = {stem: 1.0 for stem in stems_dict.keys()}
    
    # Get common length - use shape[1] for samples in (channels, samples) format
    min_length = min(audio.shape[1] if len(audio.shape) > 1 else len(audio) for audio in stems_dict.values())
    
    # Get number of channels from first stem
    first_audio = next(iter(stems_dict.values()))
    n_channels = first_audio.shape[0] if len(first_audio.shape) > 1 else 1
    
    # Mix stems together
    result = np.zeros((n_channels, min_length), dtype=np.float32)
    
    for stem_name, audio in stems_dict.items():
        weight = weights.get(stem_name, 1.0)
        if len(audio.shape) > 1:
            result += audio[:, :min_length] * weight
        else:
            result[0, :] += audio[:min_length] * weight
    
    # Find peak value in the mixed audio
    max_val = np.abs(result).max()
    
    # Normalize to prevent clipping while preserving relative dynamics
    # If mixed result would clip, scale down; otherwise preserve loudness
    if max_val > 1.0:
        # Scale to just under 1.0 for safety
        result = result / (max_val * 0.98)  # Leave 2% headroom
    
    return result.astype(np.float32)
