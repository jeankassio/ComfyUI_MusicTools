"""
ComfyUI Music Tools - Core Audio Processing Modules
"""

from .audio_repair import repair_audio
from .genre_presets import apply_genre_finish
from .limiter import apply_true_peak_limiter, measure_true_peak_db
from .stereo_enhance import apply_pseudo_stereo, apply_stereo_correlation_fix, apply_stereo_widening
from .utils import (
    apply_compression, apply_eq, apply_gain, apply_reverb, audio_to_numpy,
    calculate_lufs, enhance_stereo, get_progress_bar, mix_audio,
    normalize_to_lufs, numpy_to_audio_tensor, recombine_stems,
    restore_frequency, separate_all_stems, separate_stems,
    spectral_subtraction, trim_audio, upscale_audio,
)
from .vocal_enhance import (
    apply_breath_smoother, apply_deesser, apply_vocal_naturalizer,
    apply_vocal_reverb,
)


def process_audio_stems(*args, **kwargs):
    """Load the optional AI/mastering module only when this pipeline is used."""
    from .enhanced_master_audio import process_audio_stems as _process_audio_stems

    return _process_audio_stems(*args, **kwargs)

__all__ = [
    'audio_to_numpy',
    'numpy_to_audio_tensor',
    'spectral_subtraction',
    'upscale_audio',
    'restore_frequency',
    'enhance_stereo',
    'calculate_lufs',
    'normalize_to_lufs',
    'apply_eq',
    'apply_reverb',
    'apply_compression',
    'apply_gain',
    'mix_audio',
    'trim_audio',
    'separate_stems',
    'separate_all_stems',
    'recombine_stems',
    'get_progress_bar',
    'apply_deesser',
    'apply_breath_smoother',
    'apply_vocal_reverb',
    'apply_vocal_naturalizer',
    'apply_stereo_widening',
    'apply_stereo_correlation_fix',
    'apply_pseudo_stereo',
    'process_audio_stems',
    'repair_audio',
    'apply_genre_finish',
    'apply_true_peak_limiter',
    'measure_true_peak_db',
]
