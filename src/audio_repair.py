"""Conservative repair of invalid samples, DC offset, clipping, and clicks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


REPAIR_MODES = ("Auto (All)", "De-click Only", "De-clip Only", "DC Offset Only", "Off")


@dataclass
class RepairStats:
    nonfinite: int = 0
    clicks: int = 0
    clip_events: int = 0
    clip_samples: int = 0
    long_clips_skipped: int = 0
    dc_channels: int = 0
    max_dc_before: float = 0.0
    safety_gain_db: float = 0.0

    def report(self) -> str:
        dc_db = -120.0 if self.max_dc_before <= 1e-6 else 20.0 * np.log10(self.max_dc_before)
        return (
            f"nonfinite={self.nonfinite}; clicks={self.clicks}; "
            f"clip_events={self.clip_events}; clip_samples={self.clip_samples}; "
            f"long_clips_skipped={self.long_clips_skipped}; "
            f"dc_channels={self.dc_channels}; dc_before={dc_db:.1f} dBFS; "
            f"safety_gain={self.safety_gain_db:.2f} dB"
        )


def _canonical_audio(audio: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(audio, dtype=np.float32)
    original_ndim = array.ndim
    if original_ndim == 1:
        array = array[np.newaxis, np.newaxis, :]
    elif original_ndim == 2:
        array = array[np.newaxis, :, :]
    elif original_ndim != 3:
        raise ValueError(f"Expected [T], [C, T], or [B, C, T], got {array.shape}")
    return array.copy(), original_ndim


def _restore_shape(audio: np.ndarray, original_ndim: int) -> np.ndarray:
    if original_ndim == 1:
        return audio[0, 0]
    if original_ndim == 2:
        return audio[0]
    return audio


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _repair_nonfinite(channel: np.ndarray) -> tuple[np.ndarray, int]:
    invalid = ~np.isfinite(channel)
    count = int(np.count_nonzero(invalid))
    if count == 0:
        return channel, 0
    valid_indices = np.flatnonzero(~invalid)
    if valid_indices.size == 0:
        return np.zeros_like(channel), count
    repaired = channel.copy()
    bad_indices = np.flatnonzero(invalid)
    repaired[bad_indices] = np.interp(bad_indices, valid_indices, channel[valid_indices])
    return repaired, count


def _estimate_dc(channel: np.ndarray) -> float:
    if channel.size < 32:
        return float(np.mean(channel)) if channel.size else 0.0
    low, high = np.percentile(channel, [1.0, 99.0])
    return float(np.mean(np.clip(channel, low, high), dtype=np.float64))


def _hermite_fill(channel: np.ndarray, start: int, end: int, peak_limit: float | None = None) -> bool:
    left = start - 1
    right = end + 1
    if left < 1 or right + 1 >= channel.size:
        return False
    span = right - left
    t = np.arange(1, span, dtype=np.float64) / span
    y0 = float(channel[left])
    y1 = float(channel[right])
    m0 = float(channel[left] - channel[left - 1]) * span
    m1 = float(channel[right + 1] - channel[right]) * span
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    values = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
    if peak_limit is not None:
        values = np.clip(values, -peak_limit, peak_limit)
    channel[start : end + 1] = values.astype(np.float32)
    return True


def _repair_clipping(
    channel: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    max_clip_ms: float,
    sensitivity: float,
) -> tuple[np.ndarray, int, int, int]:
    if channel.size < 8:
        return channel, 0, 0, 0
    absolute = np.abs(channel)
    peak = float(np.max(absolute))
    if peak < 1e-6:
        return channel, 0, 0, 0

    configured_level = 10.0 ** (threshold_dbfs / 20.0)
    plateau_tolerance = max(2e-5, peak * (0.0003 + sensitivity * 0.0012))
    # The configured level qualifies the file as potentially clipped; the
    # actual mask follows only its flat top. Masking everything above the
    # threshold would include clean sine shoulders and reject the plateau.
    # Never infer clipping below the level selected by the user. The previous
    # global candidate-count exception mistook the repeated maxima of a clean
    # periodic tone for hundreds of clipping events.
    if peak < configured_level:
        return channel, 0, 0, 0
    candidate = np.abs(absolute - peak) <= plateau_tolerance
    max_samples = max(2, int(round(max_clip_ms * sample_rate / 1000.0)))
    repaired_events = 0
    repaired_samples = 0
    skipped = 0

    for start, end in _runs(candidate):
        length = end - start + 1
        # Two equal samples can occur naturally when a sampled waveform peaks
        # between them. Requiring at least three samples is intentionally
        # conservative; very short defects are still handled by de-clicking.
        if length < 3:
            continue
        if length > max_samples:
            skipped += 1
            continue
        if start < 2 or end + 2 >= channel.size:
            continue
        segment = channel[start : end + 1]
        if np.any(np.signbit(segment) != np.signbit(segment[0])):
            continue
        if float(np.ptp(np.abs(segment))) > plateau_tolerance * 2.0:
            continue

        # A real hard-clipping plateau is substantially flatter than the
        # slopes entering and leaving it. This rejects broad, smooth maxima
        # from loud low-frequency tones while retaining flat encoded plateaus.
        internal_slope = float(np.max(np.abs(np.diff(segment))))
        boundary_slope = min(
            abs(float(channel[start] - channel[start - 1])),
            abs(float(channel[end + 1] - channel[end])),
        )
        flatness_limit = max(2e-7, boundary_slope * 0.15)
        if internal_slope > flatness_limit:
            continue

        # The clean samples must approach the plateau and then move away from it.
        before_slope = absolute[start] - absolute[start - 1]
        after_slope = absolute[end + 1] - absolute[end]
        if before_slope < -plateau_tolerance or after_slope > plateau_tolerance:
            continue
        peak_limit = max(peak, float(np.max(np.abs(segment)))) * (10.0 ** (3.0 / 20.0))
        if _hermite_fill(channel, start, end, peak_limit=peak_limit):
            repaired_events += 1
            repaired_samples += length

    return channel, repaired_events, repaired_samples, skipped


def _repair_clicks(
    channel: np.ndarray,
    sample_rate: int,
    max_click_ms: float,
    sensitivity: float,
) -> tuple[np.ndarray, int]:
    max_samples = max(1, int(round(max_click_ms * sample_rate / 1000.0)))
    if channel.size < max(16, max_samples * 4):
        return channel, 0

    radius = max(3, max_samples * 3)
    window = radius * 2 + 1
    prediction = np.empty_like(channel)
    prediction[1:-1] = (channel[:-2] + channel[2:]) * 0.5
    prediction[0] = channel[0]
    prediction[-1] = channel[-1]
    residual = channel - prediction
    # Use an upper local quantile rather than a median. A median becomes zero
    # for valid high-frequency tones whose prediction residual alternates
    # between peaks and zero, causing their periodic peaks to look like clicks.
    local_scale = ndimage.percentile_filter(
        np.abs(residual), percentile=75.0, size=window, mode="reflect"
    )
    multiplier = 14.0 - 8.0 * float(np.clip(sensitivity, 0.0, 1.0))
    floor = max(1e-6, float(np.median(np.abs(residual))) * 2.0)
    candidate = np.abs(residual) > multiplier * np.maximum(local_scale, floor)

    repaired_events = 0
    for start, end in _runs(candidate):
        length = end - start + 1
        if length > max_samples or start < 2 or end + 2 >= channel.size:
            continue
        if _hermite_fill(channel, start, end):
            repaired_events += 1
    return channel, repaired_events


def repair_audio(
    audio: np.ndarray,
    sample_rate: int = 44100,
    mode: str = "Auto (All)",
    sensitivity: float = 0.5,
    clip_threshold_dbfs: float = -1.0,
    max_click_ms: float = 0.5,
    max_clip_ms: float = 10.0,
    output_ceiling_dbfs: float = -1.0,
) -> tuple[np.ndarray, str]:
    """Repair audio and return the processed signal plus a compact report."""
    if mode not in REPAIR_MODES:
        raise ValueError(f"Unknown repair mode: {mode}")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    canonical, original_ndim = _canonical_audio(audio)
    stats = RepairStats()
    if mode == "Off" or canonical.shape[-1] == 0:
        return _restore_shape(canonical, original_ndim), stats.report()

    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    do_dc = mode in ("Auto (All)", "DC Offset Only")
    do_declip = mode in ("Auto (All)", "De-clip Only")
    do_declick = mode in ("Auto (All)", "De-click Only")
    changed_batches = np.zeros(canonical.shape[0], dtype=bool)

    for batch in range(canonical.shape[0]):
        for channel_index in range(canonical.shape[1]):
            channel = canonical[batch, channel_index]
            channel, invalid_count = _repair_nonfinite(channel)
            stats.nonfinite += invalid_count
            changed_batches[batch] |= invalid_count > 0

            if do_dc:
                dc = _estimate_dc(channel)
                stats.max_dc_before = max(stats.max_dc_before, abs(dc))
                # In explicit mode, honor the request; Auto avoids altering a
                # clean signal for negligible numerical means.
                if abs(dc) >= 1e-3 or (mode == "DC Offset Only" and abs(dc) > 1e-7):
                    channel -= dc
                    stats.dc_channels += 1
                    changed_batches[batch] = True

            if do_declip:
                channel, events, samples, skipped = _repair_clipping(
                    channel,
                    sample_rate,
                    float(np.clip(clip_threshold_dbfs, -18.0, -0.01)),
                    float(np.clip(max_clip_ms, 0.2, 30.0)),
                    sensitivity,
                )
                stats.clip_events += events
                stats.clip_samples += samples
                stats.long_clips_skipped += skipped
                changed_batches[batch] |= events > 0

            if do_declick:
                channel, clicks = _repair_clicks(
                    channel,
                    sample_rate,
                    float(np.clip(max_click_ms, 0.05, 3.0)),
                    sensitivity,
                )
                stats.clicks += clicks
                changed_batches[batch] |= clicks > 0

            canonical[batch, channel_index] = channel

    if np.any(changed_batches) and canonical.size:
        ceiling = 10.0 ** (float(np.clip(output_ceiling_dbfs, -12.0, -0.01)) / 20.0)
        safety_gains = np.ones(canonical.shape[0], dtype=np.float32)
        for batch in np.flatnonzero(changed_batches):
            peak = float(np.max(np.abs(canonical[batch])))
            if peak > ceiling:
                safety_gains[batch] = ceiling / peak
                canonical[batch] *= safety_gains[batch]
        minimum_gain = float(np.min(safety_gains))
        if minimum_gain < 1.0:
            stats.safety_gain_db = float(20.0 * np.log10(minimum_gain))

    return _restore_shape(canonical.astype(np.float32), original_ndim), stats.report()
