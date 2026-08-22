"""Stereo-linked true-peak measurement and limiting utilities."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def _canonical_audio(audio: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(audio, dtype=np.float32)
    original_ndim = array.ndim
    if original_ndim == 1:
        array = array[np.newaxis, np.newaxis, :]
    elif original_ndim == 2:
        array = array[np.newaxis, :, :]
    elif original_ndim != 3:
        raise ValueError(f"Expected [T], [C, T], or [B, C, T], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Audio contains NaN or infinite values")
    return array, original_ndim


def _restore_shape(audio: np.ndarray, original_ndim: int) -> np.ndarray:
    if original_ndim == 1:
        return audio[0, 0]
    if original_ndim == 2:
        return audio[0]
    return audio


def _oversample(audio: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1 or audio.shape[-1] == 0:
        return audio.astype(np.float32, copy=True)
    padtype = "line" if audio.shape[-1] > 1 else "constant"
    return signal.resample_poly(audio, factor, 1, axis=-1, padtype=padtype).astype(np.float32)


def measure_true_peak_db(audio: np.ndarray, oversample: int = 4) -> float:
    """Return the highest oversampled peak in dBTP across all batches/channels."""
    if oversample not in (1, 2, 4, 8):
        raise ValueError("Oversample must be one of 1, 2, 4, or 8")
    canonical, _ = _canonical_audio(audio)
    if canonical.shape[-1] == 0 or not np.any(canonical):
        return -120.0
    peak = float(np.max(np.abs(_oversample(canonical, oversample))))
    return float(max(-120.0, 20.0 * np.log10(max(peak, 1e-6))))


def apply_true_peak_limiter(
    audio: np.ndarray,
    sample_rate: int = 44100,
    ceiling_db: float = -1.0,
    lookahead_ms: float = 3.0,
    release_ms: float = 80.0,
    oversample: int = 4,
) -> tuple[np.ndarray, float, float]:
    """
    Limit oversampled peaks with a stereo-linked gain envelope.

    Returns ``(audio, max_gain_reduction_db, output_true_peak_dbtp)``. Batches
    are processed independently, while channels inside each batch share the
    same envelope so the stereo image is not shifted.
    """
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if oversample not in (1, 2, 4, 8):
        raise ValueError("Oversample must be one of 1, 2, 4, or 8")
    ceiling_db = float(np.clip(ceiling_db, -24.0, -0.01))
    lookahead_ms = float(np.clip(lookahead_ms, 0.0, 20.0))
    release_ms = float(np.clip(release_ms, 5.0, 2000.0))
    ceiling = 10.0 ** (ceiling_db / 20.0)

    canonical, original_ndim = _canonical_audio(audio)
    if canonical.shape[-1] == 0:
        return _restore_shape(canonical.copy(), original_ndim), 0.0, -120.0

    output = np.empty_like(canonical, dtype=np.float32)
    max_reduction_db = 0.0

    for batch_index, batch in enumerate(canonical):
        upsampled = _oversample(batch, oversample)
        linked_peak = np.max(np.abs(upsampled), axis=0)
        input_peak = float(np.max(linked_peak)) if linked_peak.size else 0.0
        if input_peak <= ceiling:
            output[batch_index] = batch
            continue

        lookahead_samples = int(round(lookahead_ms * sample_rate * oversample / 1000.0))
        if lookahead_samples > 0:
            # A centered window is deliberately conservative: it starts gain
            # reduction before a transient and also prevents a release bump.
            window = lookahead_samples * 2 + 1
            envelope = ndimage.maximum_filter1d(linked_peak, size=window, mode="nearest")
        else:
            envelope = linked_peak

        required_gain = np.minimum(1.0, ceiling / np.maximum(envelope, 1e-12)).astype(np.float32)
        alpha = float(np.exp(-1.0 / (release_ms * sample_rate * oversample / 1000.0)))
        smoothed, _ = signal.lfilter(
            [1.0 - alpha],
            [1.0, -alpha],
            required_gain,
            zi=[alpha * required_gain[0]],
        )
        # Instant attack guarantees that smoothing can never exceed the gain
        # needed at a peak; the one-pole section controls only the release.
        gain = np.minimum(smoothed, required_gain).astype(np.float32)
        limited_up = upsampled * gain[np.newaxis, :]
        limited = signal.resample_poly(limited_up, 1, oversample, axis=-1)
        limited = limited[..., : batch.shape[-1]]
        if limited.shape[-1] < batch.shape[-1]:
            limited = np.pad(limited, ((0, 0), (0, batch.shape[-1] - limited.shape[-1])))

        # The reconstruction filter can create a small new intersample peak.
        # Measure again and apply one transparent linked safety correction.
        reconstructed_peak = float(np.max(np.abs(_oversample(limited, oversample))))
        if reconstructed_peak > ceiling:
            limited *= ceiling / reconstructed_peak

        output[batch_index] = limited.astype(np.float32)
        reduction_db = -20.0 * np.log10(max(float(np.min(gain)), 1e-12))
        max_reduction_db = max(max_reduction_db, float(reduction_db))

    output_peak_db = measure_true_peak_db(output, oversample)
    if output_peak_db > ceiling_db + 0.01:
        correction = 10.0 ** ((ceiling_db - output_peak_db) / 20.0)
        output *= correction
        max_reduction_db = max(max_reduction_db, -20.0 * np.log10(correction))
        output_peak_db = measure_true_peak_db(output, oversample)

    return (
        _restore_shape(output.astype(np.float32, copy=False), original_ndim),
        float(max_reduction_db),
        float(output_peak_db),
    )

