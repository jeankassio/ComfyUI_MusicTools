"""
ComfyUI Music Tools - A comprehensive music processing node pack for ComfyUI
"""

from .nodes import (
    Music_NoiseRemove,
    Music_AudioUpscale,
    Music_StereoEnhance,
    Music_LufsNormalizer,
    Music_Equalize,
    Music_Reverb,
    Music_Compressor,
    Music_Gain,
    Music_AudioMixer,
    Music_AudioTrimmer,
    Music_StemSeparation,
    Music_StemRecombination,
    Music_MasterAudioEnhancement,
)

NODE_CLASS_MAPPINGS = {
    "Music_NoiseRemove": Music_NoiseRemove,
    "Music_AudioUpscale": Music_AudioUpscale,
    "Music_StereoEnhance": Music_StereoEnhance,
    "Music_LufsNormalizer": Music_LufsNormalizer,
    "Music_Equalize": Music_Equalize,
    "Music_Reverb": Music_Reverb,
    "Music_Compressor": Music_Compressor,
    "Music_Gain": Music_Gain,
    "Music_AudioMixer": Music_AudioMixer,
    "Music_AudioTrimmer": Music_AudioTrimmer,
    "Music_StemSeparation": Music_StemSeparation,
    "Music_StemRecombination": Music_StemRecombination,
    "Music_MasterAudioEnhancement": Music_MasterAudioEnhancement,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Music_NoiseRemove": "Music - Noise Remove",
    "Music_AudioUpscale": "Music - Audio Upscale",
    "Music_StereoEnhance": "Music - Stereo Enhance",
    "Music_LufsNormalizer": "Music - LUFS Normalizer",
    "Music_Equalize": "Music - Equalize",
    "Music_Reverb": "Music - Reverb",
    "Music_Compressor": "Music - Compressor",
    "Music_Gain": "Music - Gain",
    "Music_AudioMixer": "Music - Audio Mixer",
    "Music_AudioTrimmer": "Music - Audio Trimmer",
    "Music_StemSeparation": "Music - Stem Separation",
    "Music_StemRecombination": "Music - Stem Recombination",
    "Music_MasterAudioEnhancement": "Music - Master Audio Enhancement",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
