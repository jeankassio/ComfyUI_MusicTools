#!/usr/bin/env python3
"""
Standalone test script for Vocal Naturalizer character presets.
No ComfyUI installation required.

Usage:
    python test_presets.py --input /path/to/vocal_stem.wav
    python test_presets.py --input /path/to/vocal_stem.wav --output-dir /tmp/out

Outputs one WAV per preset named <input_stem>_<preset_name>.wav.
Prints RMS (dBFS) and crest factor for each preset vs the dry signal.

Dependencies: numpy, scipy, soundfile
    pip install numpy scipy soundfile
    # or: conda run -n audioai python test_presets.py --input ...
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Load vocal_enhance directly to avoid src/__init__.py side-effects
# (enhanced_master_audio.py attempts to create /Users/models at import time)
# ---------------------------------------------------------------------------
def _load_vocal_enhance():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "src", "vocal_enhance.py")
    if not os.path.exists(path):
        sys.exit(f"Cannot find src/vocal_enhance.py relative to {here}")
    spec = importlib.util.spec_from_file_location("vocal_enhance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ve = _load_vocal_enhance()
NATURALIZER_PRESETS = _ve.NATURALIZER_PRESETS
apply_naturalizer_preset = _ve.apply_naturalizer_preset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rms_db(audio: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)


def crest_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    return 20.0 * np.log10((peak / (rms + 1e-10)) + 1e-10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply all Vocal Naturalizer presets to a WAV file and report metrics."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input WAV file (any sample rate, mono or stereo)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Directory for output files (default: same directory as input)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        sys.exit(f"Input file not found: {input_path}")

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    input_stem = os.path.splitext(os.path.basename(input_path))[0]

    # Load — keep as float32 mono for processing
    audio, sr = sf.read(input_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"\nInput:  {os.path.basename(input_path)}")
    print(f"Length: {len(audio)/sr:.2f}s  |  SR: {sr} Hz")
    print(f"Output: {output_dir}")
    print()
    print(f"{'Preset':<22} {'RMS (dBFS)':>12} {'Crest (dB)':>12}  {'ΔRMS':>7}  File")
    print("─" * 80)

    dry_rms = rms_db(audio)
    dry_crest = crest_db(audio)
    print(f"{'[dry]':<22} {dry_rms:>12.1f} {dry_crest:>12.1f}  {'—':>7}  —")

    for preset_name in NATURALIZER_PRESETS:
        processed = apply_naturalizer_preset(audio, sr, preset_name)

        p_rms = rms_db(processed)
        p_crest = crest_db(processed)
        delta_rms = p_rms - dry_rms

        out_name = f"{input_stem}_{preset_name}.wav"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, processed, sr, subtype="PCM_24")

        print(f"{preset_name:<22} {p_rms:>12.1f} {p_crest:>12.1f}  {delta_rms:>+6.1f}  {out_name}")

    print("─" * 80)
    print(f"\n{len(NATURALIZER_PRESETS)} files written to {output_dir}\n")


if __name__ == "__main__":
    main()
