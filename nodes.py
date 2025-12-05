"""
ComfyUI Music Tools - Node implementations for audio processing
"""

import numpy as np
import torch
from .src.utils import (
    audio_to_numpy,
    numpy_to_audio_tensor,
    spectral_subtraction,
    upscale_audio,
    restore_frequency,
    enhance_stereo,
    calculate_lufs,
    normalize_to_lufs,
    apply_eq,
    apply_reverb,
    apply_compression,
    apply_gain,
    mix_audio,
    trim_audio,
    separate_stems,
    separate_all_stems,
    recombine_stems,
    get_progress_bar,
)


class Music_NoiseRemove:
    """
    Remove noise from audio using professional noise reduction.
    Two modes:
    - Hiss Only: Remove only high-frequency chiados (8kHz+) - safe for effects
    - Full: Remove all noise with noisereduce library
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["Hiss Only", "Full Denoise"],),
                "intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "remove_noise"
    CATEGORY = "music"
    
    def remove_noise(self, audio, mode, intensity):
        """
        Remove noise from audio.
        
        Args:
            audio: ComfyUI audio dict with 'waveform' and 'sample_rate'
            mode: "Hiss Only" (remove only chiados >8kHz) or "Full Denoise"
            intensity: Noise removal intensity (0-1)
        
        Returns:
            Tuple with processed audio dict
        """
        try:
            from enhanced_master_audio import apply_denoise_hiss_only, apply_denoise_simple
            
            print(f"\n[Music_NoiseRemove] Starting noise removal")
            print(f"[Music_NoiseRemove] Mode: {mode}")

            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            print(f"[Music_NoiseRemove] Audio: {audio_np.shape}, SR={sample_rate}Hz")
            
            # Validate audio
            if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                print(f"[Music_NoiseRemove] Audio contains NaN/Inf values, skipping")
                return (audio,)
            
            # Check minimum samples
            n_channels, n_samples = audio_np.shape
            if n_samples < 2048:
                print(f"[Music_NoiseRemove] Audio too short ({n_samples} samples), skipping")
                return (audio,)
            
            # Create progress bar
            pbar = get_progress_bar(100, "Noise Removal")
            pbar.update_absolute(10)
            
            # Apply selected denoise mode
            if mode == "Hiss Only":
                print(f"[Music_NoiseRemove] Using Hiss Only mode (removes chiados >8kHz)")
                processed = apply_denoise_hiss_only(audio_np, sample_rate, intensity)
            else:  # Full Denoise
                print(f"[Music_NoiseRemove] Using Full Denoise mode (noisereduce)")
                processed = apply_denoise_simple(audio_np, sample_rate, intensity)
            
            pbar.update_absolute(90)
            
            # Validate processed audio
            if processed is None:
                print("[Music_NoiseRemove] Noise removal returned None")
                return (audio,)
            
            if np.isnan(processed).any() or np.isinf(processed).any():
                print("[Music_NoiseRemove] Processed audio contains NaN/Inf, returning original")
                return (audio,)
            
            # Convert back to ComfyUI format
            result = numpy_to_audio_tensor(processed, sample_rate)
            pbar.update_absolute(100)
            
            print(f"[Music_NoiseRemove] ✅ Noise removed successfully!")
            print(f"[Music_NoiseRemove] Mode: {mode} | Intensity: {intensity}\n")
            return (result,)
        except Exception as e:
            print(f"[Music_NoiseRemove] Error: {e}")
            import traceback
            traceback.print_exc()
            # Return original audio if processing fails
            return (audio,)


class Music_AudioUpscale:
    """
    Upscale audio to a higher sample rate.
    Optionally restore the original frequency after upscaling.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_sample_rate": ("INT", {
                    "default": 48000,
                    "min": 16000,
                    "max": 192000,
                    "step": 8000,
                }),
                "restore_original_sr": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "upscale"
    CATEGORY = "music"
    
    def upscale(self, audio, target_sample_rate, restore_original_sr):
        """
        Upscale audio to higher sample rate.
        
        Args:
            audio: ComfyUI audio dict
            target_sample_rate: Target sample rate
            restore_original_sr: If True, restore to original SR
        
        Returns:
            Tuple with upscaled audio dict
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Create progress bar
            pbar = get_progress_bar(100, "Audio Upscale")
            pbar.update_absolute(10)
            
            # Upscale
            upscaled = upscale_audio(audio_np, target_sample_rate, sample_rate)
            pbar.update_absolute(90)
            
            # Restore if requested
            if restore_original_sr:
                upscaled = restore_frequency(upscaled, sample_rate, target_sample_rate, target_sample_rate)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(upscaled, sample_rate)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_AudioUpscale: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


class Music_StereoEnhance:
    """
    Enhance stereo separation and width.
    Works best with stereo audio files.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "enhance"
    CATEGORY = "music"
    
    def enhance(self, audio, intensity):
        """
        Enhance stereo separation and width.
        
        Args:
            audio: ComfyUI audio dict
            intensity: Enhancement intensity (0-2)
        
        Returns:
            Tuple with enhanced audio dict
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Ensure stereo
            if audio_np.shape[1] == 1:
                # Convert mono to stereo
                audio_np = np.stack([audio_np[:, 0], audio_np[:, 0]], axis=1)
            
            # Apply enhancement
            enhanced = enhance_stereo(audio_np, intensity)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(enhanced, sample_rate)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_StereoEnhance: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


class Music_LufsNormalizer:
    """
    Normalize audio to target LUFS (Loudness Units relative to Full Scale).
    Ensures consistent loudness levels.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_lufs": ("FLOAT", {
                    "default": -14.0,
                    "min": -60.0,
                    "max": 0.0,
                    "step": 1.0,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("AUDIO", "CURRENT_LUFS")
    FUNCTION = "normalize"
    CATEGORY = "music"
    
    def normalize(self, audio, target_lufs):
        """
        Normalize audio to target LUFS loudness level.
        
        Args:
            audio: ComfyUI audio dict
            target_lufs: Target LUFS value
        
        Returns:
            Tuple with normalized audio dict and LUFS value
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Calculate current LUFS
            current_lufs = calculate_lufs(audio_np, sample_rate)
            
            # Normalize
            processed = normalize_to_lufs(audio_np, target_lufs, sample_rate)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(processed, sample_rate)
            
            return (result, float(current_lufs))
        except Exception as e:
            print(f"Error in Music_LufsNormalizer: {e}")
            import traceback
            traceback.print_exc()
            return (audio, -20.0)


class Music_Equalize:
    """
    Apply parametric EQ to audio.
    Adjust frequency response using multiple bands.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "bass_freq": ("INT", {
                    "default": 100,
                    "min": 20,
                    "max": 500,
                    "step": 10,
                }),
                "bass_gain": ("FLOAT", {
                    "default": 0.0,
                    "min": -12.0,
                    "max": 12.0,
                    "step": 0.5,
                }),
                "mid_freq": ("INT", {
                    "default": 1000,
                    "min": 200,
                    "max": 5000,
                    "step": 100,
                }),
                "mid_gain": ("FLOAT", {
                    "default": 0.0,
                    "min": -12.0,
                    "max": 12.0,
                    "step": 0.5,
                }),
                "treble_freq": ("INT", {
                    "default": 10000,
                    "min": 2000,
                    "max": 20000,
                    "step": 500,
                }),
                "treble_gain": ("FLOAT", {
                    "default": 0.0,
                    "min": -12.0,
                    "max": 12.0,
                    "step": 0.5,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "equalize"
    CATEGORY = "music"
    
    def equalize(self, audio, bass_freq, bass_gain, mid_freq, mid_gain, treble_freq, treble_gain):
        """
        Apply parametric EQ to audio.
        
        Args:
            audio: ComfyUI audio dict
            bass_freq, mid_freq, treble_freq: Center frequencies
            bass_gain, mid_gain, treble_gain: Gains in dB
        
        Returns:
            Tuple with equalized audio dict
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Apply EQ
            frequencies = [bass_freq, mid_freq, treble_freq]
            gains = [bass_gain, mid_gain, treble_gain]
            
            processed = apply_eq(audio_np, frequencies, gains, sample_rate)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(processed, sample_rate)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_Equalize: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


class Music_Reverb:
    """
    Apply reverb effect to audio.
    Simulates room acoustics and space.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "decay": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "apply_reverb_effect"
    CATEGORY = "music"
    
    def apply_reverb_effect(self, audio, decay):
        """
        Apply reverb effect to audio.
        
        Args:
            audio: ComfyUI audio dict
            decay: Reverb decay factor (0-1)
        
        Returns:
            Tuple with reverb-applied audio dict
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Apply reverb
            processed = apply_reverb(audio_np, decay, sample_rate)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(processed, sample_rate)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_Reverb: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


class Music_Compressor:
    """
    Apply dynamic range compression to audio.
    Reduces the dynamic range of the audio signal.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "ratio": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 16.0,
                    "step": 0.5,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "compress"
    CATEGORY = "music"
    
    def compress(self, audio, threshold, ratio):
        """
        Apply dynamic range compression to audio.
        
        Args:
            audio: Audio data in any ComfyUI format
            threshold: Compression threshold (0-1)
            ratio: Compression ratio
        
        Returns:
            Tuple with compressed audio dict
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Apply compression
            processed = apply_compression(audio_np, threshold, ratio, sample_rate)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(processed, sample_rate)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_Compressor: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


class Music_Gain:
    """
    Apply gain (volume adjustment) to audio.
    Adjust the overall volume level in dB.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "gain_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -24.0,
                    "max": 24.0,
                    "step": 0.5,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "apply_gain_effect"
    CATEGORY = "music"
    
    def apply_gain_effect(self, audio, gain_db):
        """
        Apply gain (volume adjustment) to audio.
        
        Args:
            audio: Audio data in any ComfyUI format
            gain_db: Gain in dB
        
        Returns:
            Tuple with gain-applied audio tensor
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Apply gain
            processed = apply_gain(audio_np, gain_db)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(processed, sample_rate)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_Gain: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


class Music_AudioMixer:
    """
    Mix multiple audio samples together.
    Combines two audio streams with automatic level adjustment.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "mix"
    CATEGORY = "music"
    
    def mix(self, audio_1, audio_2):
        """
        Mix multiple audio samples together.
        
        Args:
            audio_1: First ComfyUI audio dict
            audio_2: Second ComfyUI audio dict
        
        Returns:
            Tuple with mixed audio dict
        """
        try:
            # Convert to numpy
            audio_np_1, sr1 = audio_to_numpy(audio_1)
            audio_np_2, sr2 = audio_to_numpy(audio_2)
            
            # Mix (use first sample rate)
            processed = mix_audio(audio_np_1, audio_np_2)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(processed, sr1)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_AudioMixer: {e}")
            import traceback
            traceback.print_exc()
            return (audio_1,)


class Music_AudioTrimmer:
    """
    Trim audio to a specific time range.
    Extract a portion of audio between start and end times.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                }),
                "end_time": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "trim"
    CATEGORY = "music"
    
    def trim(self, audio, start_time, end_time):
        """
        Trim audio to a specific time range.
        
        Args:
            audio: ComfyUI audio dict
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            Tuple with trimmed audio dict
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Trim
            processed = trim_audio(audio_np, start_time, end_time, sample_rate)
            
            # Convert back to dict
            result = numpy_to_audio_tensor(processed, sample_rate)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_AudioTrimmer: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


class Music_StemSeparation:
    """
    Separate audio into stems (vocals, drums, bass, music, others).
    Extract all individual components from mixed audio simultaneously.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("VOCALS", "DRUMS", "BASS", "MUSIC", "OTHERS")
    FUNCTION = "separate"
    CATEGORY = "music"
    
    def separate(self, audio):
        """
        Separate audio into all stems simultaneously.
        
        Args:
            audio: ComfyUI audio dict
        
        Returns:
            Tuple with separated stems (vocals, drums, bass, music, others)
        """
        try:
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Create progress bar
            pbar = get_progress_bar(100, "Stem Separation")
            pbar.update_absolute(10)
            
            # Separate all stems using separate_all_stems
            stems_dict = separate_all_stems(audio_np, sample_rate)
            pbar.update_absolute(90)
            
            # Convert each stem back to dict
            vocals_result = numpy_to_audio_tensor(stems_dict["vocals"], sample_rate)
            drums_result = numpy_to_audio_tensor(stems_dict["drums"], sample_rate)
            bass_result = numpy_to_audio_tensor(stems_dict["bass"], sample_rate)
            music_result = numpy_to_audio_tensor(stems_dict["music"], sample_rate)
            others_result = numpy_to_audio_tensor(stems_dict["others"], sample_rate)
            pbar.update_absolute(100)
            
            return (vocals_result, drums_result, bass_result, music_result, others_result)
        except Exception as e:
            print(f"Error in Music_StemSeparation: {e}")
            import traceback
            traceback.print_exc()
            # Return original audio for all stems
            return (audio, audio, audio, audio, audio)


class Music_StemRecombination:
    """
    Recombine separated stems back into a full mix.
    Remix processed stems with individual volume control.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vocals": ("AUDIO",),
                "drums": ("AUDIO",),
                "bass": ("AUDIO",),
                "music": ("AUDIO",),
                "others": ("AUDIO",),
                "vocals_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "drums_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "bass_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "music_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "others_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "recombine"
    CATEGORY = "music"
    
    def recombine(self, vocals, drums, bass, music, others, vocals_volume, drums_volume, bass_volume, music_volume, others_volume):
        """
        Recombine separated stems back into a full mix.
        
        Args:
            vocals, drums, bass, music, others: ComfyUI audio dicts
            vocals_volume, drums_volume, bass_volume, music_volume, others_volume: Volume multipliers
        Returns:
            Tuple with recombined audio dict
        """
        try:
            # Create progress bar
            pbar = get_progress_bar(100, "Stem Recombination")
            pbar.update_absolute(10)
            
            # Convert to numpy
            vocals_np, sr_vocals = audio_to_numpy(vocals)
            drums_np, sr_drums = audio_to_numpy(drums)
            bass_np, sr_bass = audio_to_numpy(bass)
            music_np, sr_music = audio_to_numpy(music)
            others_np, sr_others = audio_to_numpy(others)
            pbar.update_absolute(30)
            
            # Create stems dictionary
            stems_dict = {
                "vocals": vocals_np,
                "drums": drums_np,
                "bass": bass_np,
                "music": music_np,
                "others": others_np,
            }
            
            # Create weights dictionary
            weights = {
                "vocals": vocals_volume,
                "drums": drums_volume,
                "bass": bass_volume,
                "music": music_volume,
                "others": others_volume,
            }
            
            # Recombine stems
            processed = recombine_stems(stems_dict, weights)
            pbar.update_absolute(90)
            
            # Convert back to dict (use first sample rate)
            result = numpy_to_audio_tensor(processed, sr_vocals)
            pbar.update_absolute(100)
            
            return (result,)
        except Exception as e:
            print(f"Error in Music_StemRecombination: {e}")
            import traceback
            traceback.print_exc()
            return (vocals,)


class Music_MasterAudioEnhancement:
    """
    Professional master audio enhancement node.
    Improves muddy, unclear, uneven AI-generated audio to studio quality.
    
    Applies:
    - Noise reduction (spectral subtraction)
    - EQ (clarity and presence boost)
    - Compression (tame dynamics, improve clarity)
    - Normalization (LUFS standard)
    - Limiting (prevent clipping)
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                # Denoise mode selection
                "denoise_mode": (["Hiss Only", "Full Denoise", "Off"],),
                # Optional neural enhancer (SpeechBrain MetricGAN+)
                "ai_enhance": ("BOOLEAN", {"default": False}),
                "ai_mix": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                # Denoise controls
                "denoise_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                # EQ parameters (dB)
                "eq_low_gain": ("FLOAT", {
                    "default": 0.0,
                    "min": -12.0,
                    "max": 12.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "eq_mid_gain": ("FLOAT", {
                    "default": 0.5,
                    "min": -12.0,
                    "max": 12.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "eq_high_gain": ("FLOAT", {
                    "default": 1.5,
                    "min": -12.0,
                    "max": 12.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                # Clarity enhancement
                "clarity_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                # Loudness normalization (LUFS)
                "target_loudness": ("FLOAT", {
                    "default": -6.0,
                    "min": -30.0,
                    "max": 0.0,
                    "step": 1.0,
                    "display": "slider",
                }),
                # VOCAL ENHANCEMENT (Ace-Step vocals focus)
                "vocal_enhance": ("BOOLEAN", {"default": True}),
                "deesser_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "breath_smooth": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "reverb_amount": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                # VOCAL NATURALIZER (Remove auto-tune/robotic artifacts)
                "naturalize_vocal": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "enhance"
    CATEGORY = "music"
    
    def enhance(self, audio, denoise_mode, ai_enhance, ai_mix, denoise_intensity, eq_low_gain, eq_mid_gain, eq_high_gain,
                clarity_amount, target_loudness, vocal_enhance, deesser_amount, breath_smooth, reverb_amount, naturalize_vocal):
        """
        Apply professional master enhancement to audio with stem separation
        
        Args:
            audio: ComfyUI audio dict
            denoise_mode: "Hiss Only", "Full Denoise", or "Off"
            denoise_intensity: Denoise strength (0-1)
            eq_*_gain: EQ gains in dB
            clarity_amount: Clarity enhancement amount
            target_loudness: Target loudness in LUFS
            vocal_enhance: Enable vocal-specific enhancements
            deesser_amount: De-esser strength
            breath_smooth: Breath smoother strength
            reverb_amount: Reverb amount
            naturalize_vocal: Remove auto-tune/robotic artifacts (0-1)
        
        Returns:
            Enhanced audio dict
        """
        try:
            # Import here to avoid issues at module load
            from .src.enhanced_master_audio import (
                process_audio_stems,
            )
            
            # Convert to numpy
            audio_np, sample_rate = audio_to_numpy(audio)
            
            # Create progress bar
            pbar = get_progress_bar(100, "Master Enhancement")
            pbar.update_absolute(5)
            print("[Master Audio] Starting processing...")
            print(f"[Master Audio] Denoise mode: {denoise_mode}")
            pbar.update_absolute(10)
            
            # Process audio directly (no stem separation)
            print("[Master Audio] Processing audio (this will show detailed progress)...")
            audio_np = process_audio_stems(
                audio_np, sample_rate,
                eq_low_gain, eq_mid_gain, eq_high_gain,
                clarity_amount, target_loudness,
                denoise_mode, denoise_intensity,
                ai_enhance, ai_mix,
                vocal_enhance, deesser_amount, breath_smooth, reverb_amount, naturalize_vocal
            )
            pbar.update_absolute(95)
            print("[Master Audio] ✓ Processing complete!")
            
            # Convert back to dict
            result = numpy_to_audio_tensor(audio_np, sample_rate)
            pbar.update_absolute(100)
            
            return (result,)
        except Exception as e:
            print(f"Error in Master_Audio_Enhancement: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)
