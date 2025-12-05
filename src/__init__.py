"""
ComfyUI Music Tools - Core Audio Processing Modules
"""

from .utils import *
from .vocal_enhance import *
from .enhanced_master_audio import *
from .master_audio import *
from .stereo_enhance import *
from .config import *

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
    'process_audio_stems',
]
