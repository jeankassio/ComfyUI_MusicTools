#!/usr/bin/env python3
"""
Enhanced Master Audio with per-stem processing
Separates audio into stems and applies optimized processing to each
"""

import numpy as np
from scipy import signal
import os
import tempfile

# Heavy AI dependencies are imported only if ai_enhance is enabled. This keeps
# normal DSP nodes fast to load and avoids startup failures from a mismatched
# torch/torchaudio/SpeechBrain installation.
torch = None
torchaudio = None
SpectralMaskEnhancement = None
TORCHAUDIO_AVAILABLE = False
SPEECHBRAIN_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("[Denoise] WARNING: noisereduce not available, denoise will be disabled")

try:
    from huggingface_hub import snapshot_download
    HF_SNAPSHOT_AVAILABLE = True
except Exception:
    snapshot_download = None
    HF_SNAPSHOT_AVAILABLE = False

def _get_model_dir():
    """Resolve the model cache lazily, without filesystem writes at import."""
    try:
        import folder_paths

        models_root = folder_paths.models_dir
    except ImportError:
        models_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    model_dir = os.path.join(models_root, "MusicEnhance")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

_METRICGAN_ENHANCER = None

def apply_denoise_hiss_only(audio, sample_rate=44100, intensity=0.5):
    """
    Remove ONLY high-frequency hiss/noise (chiados) without affecting effects.
    Vectorized spectral subtraction above 8 kHz with safety floors.
    intensity: 0-1
    """
    try:
        # Handle multichannel by channel-wise processing
        if len(audio.shape) > 1:
            result = np.zeros_like(audio)
            for ch in range(audio.shape[0]):
                result[ch, :] = apply_denoise_hiss_only(audio[ch, :], sample_rate, intensity)
            return result

        if intensity < 0.01 or len(audio) < 1024:
            return audio

        # STFT
        nperseg = min(1024, len(audio) // 4)
        if nperseg < 512:
            return audio

        f, t, Zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # Target high frequencies (>8kHz)
        hiss_bin = np.argmax(f >= 8000)
        if hiss_bin <= 0 or hiss_bin >= len(f):
            return audio

        noise_frames = max(1, len(t) // 8)

        # Baseline noise for high band (vectorized)
        noise_levels = np.mean(mag[hiss_bin:, :noise_frames], axis=1, keepdims=True)

        # Reduction and floors
        reduction = noise_levels * (intensity * 0.6)
        floor = noise_levels * (0.3 + 0.2 * intensity)  # 30-50% floor

        cleaned_high = np.maximum(mag[hiss_bin:, :] - reduction, floor)

        # Blend only high band; keep low band untouched
        blend = 0.2 + (intensity * 0.5)  # 0.2-0.7
        mag_clean = mag.copy()
        mag_clean[hiss_bin:, :] = mag[hiss_bin:, :] * (1 - blend) + cleaned_high * blend

        # Reconstruct
        Zxx_clean = mag_clean * np.exp(1j * phase)
        _, audio_clean = signal.istft(Zxx_clean, fs=sample_rate, nperseg=nperseg)

        # Match length
        if len(audio_clean) > len(audio):
            audio_clean = audio_clean[:len(audio)]
        elif len(audio_clean) < len(audio):
            audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))

        return audio_clean.astype(np.float32)
    except Exception as e:
        print(f"[Denoise Hiss] Error: {e}")
        return audio

# Denoise function for pre-processing using noisereduce
def apply_denoise_simple(audio, sample_rate=44100, intensity=0.5):
    """
    Professional denoise using noisereduce library
    Uses stationary noise reduction algorithm
    intensity: 0-1, controls how much noise to remove
    """
    if not NOISEREDUCE_AVAILABLE or intensity < 0.01:
        return audio
    
    try:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            return np.stack(
                [apply_denoise_simple(item, sample_rate, intensity) for item in audio],
                axis=0,
            )

        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,
            prop_decrease=float(np.clip(intensity, 0.0, 1.0)),
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
        )
        return reduced.astype(np.float32)
            
    except Exception as e:
        print(f"[Denoise] Error with noisereduce: {e}, returning original audio")
        return audio


def _load_metricgan_enhancer():
    """Load SpeechBrain MetricGAN+ model into MusicEnhance cache dir."""
    global _METRICGAN_ENHANCER, torch, torchaudio
    global SpectralMaskEnhancement, TORCHAUDIO_AVAILABLE, SPEECHBRAIN_AVAILABLE

    if _METRICGAN_ENHANCER is not None:
        return _METRICGAN_ENHANCER

    if not SPEECHBRAIN_AVAILABLE:
        try:
            import torch as torch_module
            import torchaudio as torchaudio_module

            torch = torch_module
            torchaudio = torchaudio_module
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: []
            try:
                from speechbrain.inference.enhancement import SpectralMaskEnhancement as enhancer_class
            except ImportError:
                from speechbrain.pretrained import SpectralMaskEnhancement as enhancer_class
            SpectralMaskEnhancement = enhancer_class
            TORCHAUDIO_AVAILABLE = True
            SPEECHBRAIN_AVAILABLE = True
        except Exception as error:
            print(f"[AI Enhance] Optional dependencies unavailable: {error}")
            return None

    try:
        target_dir = os.path.join(_get_model_dir(), "metricgan-plus-voicebank")
        os.makedirs(target_dir, exist_ok=True)

        # Prefer a copy-based download to avoid symlink permission issues on Windows
        if HF_SNAPSHOT_AVAILABLE and snapshot_download is not None:
            try:
                downloaded_dir = snapshot_download(
                    repo_id="speechbrain/metricgan-plus-voicebank",
                    local_dir=target_dir,
                )
                target_dir = downloaded_dir
            except Exception as dl_err:
                print(f"[AI Enhance] snapshot_download warning: {dl_err}")

        # Load from the local folder (works whether files were copied or cached)
        _METRICGAN_ENHANCER = SpectralMaskEnhancement.from_hparams(
            source=target_dir,
            savedir=target_dir,
        )
        print(f"[AI Enhance] Loaded MetricGAN+ model at {target_dir}")
        return _METRICGAN_ENHANCER
    except Exception as e:
        print(f"[AI Enhance] Failed to load MetricGAN+: {e}")
        _METRICGAN_ENHANCER = None
        return None


def apply_ai_enhance(audio, sample_rate=44100, mix=0.6):
    """
    Optional neural enhancement using SpeechBrain MetricGAN+ (Hugging Face).
    - Model cached under ComfyUI/models/MusicEnhance/metricgan-plus-voicebank
    - Resamples to 16k for the model, then back to original SR
    - mix (0-1): blend of enhanced signal
    """
    enhancer = _load_metricgan_enhancer()
    if enhancer is None:
        return audio

    target_sr = 16000

    def _resample(channel, src_sr, dst_sr, target_len=None):
        resampled = signal.resample(channel, int(len(channel) * dst_sr / src_sr))
        if target_len is not None and len(resampled) != target_len:
            resampled = signal.resample(resampled, target_len)
        return resampled.astype(np.float32)

    def _process_channel(channel):
        try:
            original_len = len(channel)
            work = channel.astype(np.float32)
            if sample_rate != target_sr:
                work = _resample(work, sample_rate, target_sr)

            tensor = torch.from_numpy(work).unsqueeze(0)
            with torch.no_grad():
                enhanced_tensor = enhancer.enhance_batch(
                    tensor,
                    lengths=torch.tensor([1.0], dtype=torch.float32),
                )[0]

            enhanced = enhanced_tensor.squeeze(0).cpu().numpy()
            if sample_rate != target_sr:
                enhanced = _resample(enhanced, target_sr, sample_rate, target_len=original_len)
            elif len(enhanced) != original_len:
                enhanced = _resample(enhanced, target_sr, target_sr, target_len=original_len)
            return enhanced.astype(np.float32)
        except Exception as e:
            print(f"[AI Enhance] Channel failed: {e}")
            return channel

    try:
        if len(audio.shape) > 1:
            enhanced = np.zeros_like(audio, dtype=np.float32)
            for ch in range(audio.shape[0]):
                enhanced[ch] = _process_channel(audio[ch])
        else:
            enhanced = _process_channel(audio)

        mix = float(np.clip(mix, 0.0, 1.0))
        return (audio * (1.0 - mix) + enhanced * mix).astype(np.float32)
    except Exception as e:
        print(f"[AI Enhance] Error: {e}")
        return audio



# ============================================================================
# REMOVED UNUSED FUNCTIONS (2025-12-04)
# - apply_denoise_stem() - Unused, use apply_denoise_simple() instead
# - ensure_demucs_mdx_extra() - Stem separation removed, no longer needed
# - separate_audio_stems() - Stem separation removed, direct processing now
# - get_optimized_gains() - Per-stem processing removed
# - apply_stem_processing() - Per-stem processing removed
# ============================================================================

def process_audio_stems(audio, sample_rate, eq_low_gain, eq_mid_gain, eq_high_gain,
                        clarity_amount, target_loudness, denoise_mode="Hiss Only", denoise_intensity=0.5,
                        ai_enhance=False, ai_mix=0.6,
                        vocal_enhance=False, deesser_amount=0.5, breath_smooth=0.3, reverb_amount=0.2, naturalize_vocal=0.5):
    """
    Direct processing pipeline WITHOUT stem separation.
    Applies all enhancements directly to the original audio.
    This preserves the original mix cohesion and dynamics!
    
    Args:
        denoise_mode: "Hiss Only", "Full Denoise", or "Off"
        denoise_intensity: 0-1 for denoise strength
        ai_enhance: bool, optional neural enhancement via SpeechBrain MetricGAN+
        ai_mix: blend of enhanced signal (0-1)
        vocal_enhance: bool, enable vocal-focused processing (de-esser, breath smoother, reverb)
        deesser_amount: 0-1, sibilance reduction strength
        breath_smooth: 0-1, breath/transition smoothing
        reverb_amount: 0-1, reverb amount for vocal cohesion
        naturalize_vocal: 0-1, remove auto-tune/robotic artifacts from AI vocals
    """
    print("\n" + "="*60)
    print("  MASTER AUDIO ENHANCEMENT - Direct Processing (No Separation)")
    print("="*60 + "\n")

    # Ensure stereo
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio])

    # STEP 0: Optional AI enhancement
    if ai_enhance:
        print("[0/3] AI ENHANCEMENT - SpeechBrain MetricGAN+")
        print("-" * 60)
        print(f"[AI Enhance] Processing with MetricGAN+ (mix {ai_mix:.2f})...", end='', flush=True)
        try:
            audio = apply_ai_enhance(audio, sample_rate, ai_mix)
            if _METRICGAN_ENHANCER is None:
                print(" skipped (optional AI stack unavailable).\n")
            else:
                print(" done.\n")
        except Exception as e:
            print(f" failed: {e}\n")
    else:
        print("[0/3] AI ENHANCEMENT - Skipped\n")

    # STEP 1: DENOISE (if enabled)
    if denoise_mode != "Off" and denoise_intensity > 0.01:
        print(f"[1/3] DENOISING - Mode: {denoise_mode}")
        print("-" * 60)
        print(f"[Denoise] Cleaning audio ({denoise_intensity*100:.0f}%)...", end='', flush=True)
        try:
            if denoise_mode == "Hiss Only":
                if audio.shape[0] == 2:
                    audio[0] = apply_denoise_hiss_only(audio[0], sample_rate, denoise_intensity)
                    audio[1] = apply_denoise_hiss_only(audio[1], sample_rate, denoise_intensity)
                else:
                    audio = apply_denoise_hiss_only(audio, sample_rate, denoise_intensity)
            else:  # Full Denoise
                if audio.shape[0] == 2:
                    audio[0] = apply_denoise_simple(audio[0], sample_rate, denoise_intensity)
                    audio[1] = apply_denoise_simple(audio[1], sample_rate, denoise_intensity)
                else:
                    audio = apply_denoise_simple(audio, sample_rate, denoise_intensity)
            print(" done.\n")
        except Exception as e:
            print(f" failed: {e}\n")
    else:
        print("[1/3] DENOISING - Skipped (Mode: Off)\n")

    # STEP 2: GLOBAL PROCESSING
    print("[2/3] APPLYING MASTER PROCESSING")
    print("-" * 60)
    print("[Master] Processing audio...", end='', flush=True)
    
    # Apply processing directly to the original audio (no separation)
    try:
        from .master_audio import (
            apply_parametric_eq, 
            apply_clarity_enhancement, 
            apply_loudness_normalization,
            apply_multiband_compression
        )
        from .stereo_enhance import apply_stereo_widening, apply_stereo_correlation_fix
        
        # 1. Fix stereo phase issues
        if audio.shape[0] == 2:
            print("\n[Master 1/6] Checking stereo phase...", end='', flush=True)
            audio = apply_stereo_correlation_fix(audio, sample_rate)
        
        # 2. EQ shaping
        print("\n[Master 2/6] Applying EQ...", end='', flush=True)
        audio = apply_parametric_eq(
            audio,
            low_gain=eq_low_gain,
            mid_gain=eq_mid_gain,
            high_gain=eq_high_gain,
            sample_rate=sample_rate
        )
        
        # 3. Multiband compression for transparent dynamics
        print("\n[Master 3/6] Multiband compression...", end='', flush=True)
        audio = apply_multiband_compression(
            audio,
            sample_rate=sample_rate,
            threshold=0.35,
            ratio=3.5,
            attack_ms=5,
            release_ms=80
        )
        
        # 4. Clarity enhancement (transient shaper + harmonic exciter + presence)
        print("\n[Master 4/6] Clarity enhancement...", end='', flush=True)
        audio = apply_clarity_enhancement(
            audio,
            clarity_amount=clarity_amount,
            sample_rate=sample_rate
        )
        
        # 4.5 VOCAL ENHANCEMENT (optional: de-esser, breath smoother, reverb)
        if vocal_enhance and max(deesser_amount, breath_smooth, reverb_amount, naturalize_vocal) > 0.01:
            try:
                from .vocal_enhance import apply_deesser, apply_breath_smoother, apply_vocal_reverb, apply_vocal_naturalizer
                print("\n[Master 4.5/6] Vocal enhancement...", end='', flush=True)
                
                # Vocal naturalizer: remove auto-tune/robotic artifacts (FIRST)
                if naturalize_vocal > 0.01:
                    if audio.shape[0] == 2:
                        audio[0] = apply_vocal_naturalizer(audio[0], sample_rate, naturalize_vocal)
                        audio[1] = apply_vocal_naturalizer(audio[1], sample_rate, naturalize_vocal)
                    else:
                        audio = apply_vocal_naturalizer(audio, sample_rate, naturalize_vocal)
                
                # De-esser: remove sibilance
                if audio.shape[0] == 2:
                    audio[0] = apply_deesser(audio[0], sample_rate, deesser_amount)
                    audio[1] = apply_deesser(audio[1], sample_rate, deesser_amount)
                else:
                    audio = apply_deesser(audio, sample_rate, deesser_amount)
                
                # Breath smoother: smooth transitions
                if breath_smooth > 0.01:
                    if audio.shape[0] == 2:
                        audio[0] = apply_breath_smoother(audio[0], sample_rate, breath_smooth)
                        audio[1] = apply_breath_smoother(audio[1], sample_rate, breath_smooth)
                    else:
                        audio = apply_breath_smoother(audio, sample_rate, breath_smooth)
                
                # Reverb: add space
                if reverb_amount > 0.01:
                    if audio.shape[0] == 2:
                        audio[0] = apply_vocal_reverb(audio[0], sample_rate, reverb_amount, "small_room")
                        audio[1] = apply_vocal_reverb(audio[1], sample_rate, reverb_amount, "small_room")
                    else:
                        audio = apply_vocal_reverb(audio, sample_rate, reverb_amount, "small_room")
            except (ImportError, Exception) as e:
                print(f"\n[Master] Vocal enhancement skipped: {e}")
        
        # 5. Stereo widening for spaciousness
        print("\n[Master 5/6] Stereo widening...", end='', flush=True)
        if audio.shape[0] == 2:
            width = 1.0 + (clarity_amount * 0.3)  # Link width to clarity
            audio = apply_stereo_widening(audio, width, sample_rate)
        
        # 6. Loudness normalization (ITU-R BS.1770-4)
        print("\n[Master 6/6] Loudness normalization...", end='', flush=True)
        audio = apply_loudness_normalization(
            audio,
            target_loudness=target_loudness,
            sample_rate=sample_rate
        )
    except (ImportError, ModuleNotFoundError) as e:
        print(f"\nWarning: {e}")
        # Fallback: just normalize
        max_val = np.abs(audio).max()
        if max_val > 0.98:
            audio = audio * (0.98 / max_val)
    
    print("\n[Master] Processing complete.\n")
    
    # STEP 3: FINAL LIMITING
    print("[3/3] APPLYING FINAL LIMITER")
    print("-" * 60)
    print("[Limiter] Protecting against clipping...", end='', flush=True)
    
    try:
        from .master_audio import apply_soft_limiter
        audio = apply_soft_limiter(audio, threshold=0.98, lookahead_ms=2.0, sample_rate=sample_rate)
    except Exception as e:
        print(f"\nLimiter warning: {e}")
        # Simple fallback limiter
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio * (0.99 / max_val)
    
    print(" done.\n")
    print("="*60)
    print("  Mastering finished! Ready to export")
    print("="*60 + "\n")
    
    return audio
