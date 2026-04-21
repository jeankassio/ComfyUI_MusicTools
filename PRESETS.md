# Vocal Naturalizer — Character Presets

Six character presets for `apply_vocal_naturalizer()`, each tuned to a specific
sonic aesthetic. Every preset sets the five independent DSP parameters directly,
so they can be used as starting points for further fine-tuning.

| Preset | Character | Best for | Known limitations |
|---|---|---|---|
| `cold_spoken` | Dry, controlled, minimal processing | Podcast narration, audiobook, voice-over, ASMR | Very low noise masking may leave faint pitch-quantization steps audible on sustained vowels |
| `krautrock_motorik` | Robotic precision with long transitions | Electronic / industrial beat vocals, motorik-paced delivery | Sibilant resonance on "s" and "sh" sounds — saturation uniformity is a structural limit of the current DSP chain; analog-style dynamic variation requires a future amplitude-modulated distortion module |
| `industrial_80s` | Heavy formant modulation, coarse noise texture | Aggressive industrial, EBM, power electronics | Saturation uniformity is a structural limit of the current DSP chain; analog-style dynamic variation requires a future amplitude-modulated distortion module |
| `trip_hop_urban` | Warm metallic cut, smooth transitions | Trip-hop, lo-fi hip-hop, urban spoken word | Formant modulation at 0.50 can slightly colour long held notes |
| `warm_documentary` | Natural, open, high formant depth | Documentary narration, cinematic voice-over, jazz vocal | Slight low-end softening from high transition_smooth (0.80); may need a bass shelf to compensate |
| `raw_untouched` | No humanization — full metallic cut only | Reference / bypass comparison, harsh-noise aesthetics | Maximum metallic_cut (1.0) will cut all 6–10 kHz content; not suitable for music with natural high-frequency content |

## Parameters

Each preset is a dictionary with five keys in the range `[0.0, 1.0]`:

| Parameter | Controls |
|---|---|
| `pitch_drift` | Vibrato depth and phase-modulation intensity |
| `formant_variation` | Random formant modulation depth (200–3000 Hz band) |
| `metallic_cut` | Attenuation of 6–10 kHz metallic digital artifacts |
| `noise_masking` | Shaped noise (1–4 kHz) to mask pitch-quantization steps |
| `transition_smooth` | LP (50 Hz) smoothing on pitch transitions |

## Usage

```python
from src.vocal_enhance import apply_naturalizer_preset, NATURALIZER_PRESETS

# Apply a named preset
audio_out = apply_naturalizer_preset(audio_np, sample_rate, "warm_documentary")

# Or call apply_vocal_naturalizer directly with individual parameters
from src.vocal_enhance import apply_vocal_naturalizer
audio_out = apply_vocal_naturalizer(audio_np, sample_rate,
    pitch_drift=0.45, formant_variation=0.45,
    metallic_cut=0.50, noise_masking=0.15, transition_smooth=0.80)

# Legacy single-parameter call still works (maps amount → all five params)
audio_out = apply_vocal_naturalizer(audio_np, sample_rate, amount=0.5)
```

## Validation

All presets validated on Suno v4/v5 spoken word stems using Google AI Studio
audio analysis. Source material: Italian/Tagalog spoken word
(Valter Ego, "Tuttə fannœ tutto", 2026).
