from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))

import ComfyUI_MusicTools as package
from ComfyUI_MusicTools.nodes import (
    Music_AudioMixer,
    Music_AudioRepair,
    Music_AudioUpscale,
    Music_Fix,
    Music_NoiseRemove,
    Music_StereoEnhance,
)
from ComfyUI_MusicTools.src.audio_repair import repair_audio
from ComfyUI_MusicTools.src.genre_presets import (
    GENRE_OPTIONS,
    GENRE_TO_PROFILE,
    PROFILES,
    _adaptive_compress,
)
from ComfyUI_MusicTools.src.limiter import apply_true_peak_limiter, measure_true_peak_db
from ComfyUI_MusicTools.src.master_audio import apply_multiband_compression
from ComfyUI_MusicTools.src.utils import (
    apply_eq,
    apply_gain,
    audio_to_numpy,
    calculate_lufs,
    mix_audio,
    normalize_to_lufs,
    recombine_stems,
    separate_all_stems,
)


def make_audio(
    sample_rate: int = 48000,
    seconds: float = 1.0,
    channels: int = 1,
    batches: int = 1,
    amplitude: float = 0.1,
    frequency: float = 997.0,
):
    time = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    mono = amplitude * np.sin(2.0 * np.pi * frequency * time)
    waveform = np.tile(mono, (batches, channels, 1)).astype(np.float32)
    return {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}


class FormatAndNodeTests(unittest.TestCase):
    def test_registry_and_node_catalog_are_in_sync(self):
        catalog = json.loads((REPO_ROOT / "node_list.json").read_text(encoding="utf-8"))
        self.assertIsInstance(catalog, dict)
        self.assertEqual(set(catalog), set(package.NODE_CLASS_MAPPINGS))
        self.assertEqual(len(catalog), 15)

    def test_audio_conversion_preserves_batches(self):
        source = make_audio(batches=3, channels=2)
        array, sample_rate = audio_to_numpy(source)
        self.assertEqual(array.shape, (3, 2, 48000))
        self.assertEqual(sample_rate, 48000)

    def test_audio_conversion_does_not_alias_input_tensor(self):
        source = make_audio(seconds=0.1)
        original_sample = source["waveform"][0, 0, 0].item()
        array, _ = audio_to_numpy(source)
        array[0, 0, 0] = 0.75
        self.assertEqual(source["waveform"][0, 0, 0].item(), original_sample)

    def test_upscale_sets_the_output_sample_rate(self):
        source = make_audio(sample_rate=24000, seconds=0.25, batches=2)
        result = Music_AudioUpscale().upscale(source, 48000, False)[0]
        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(tuple(result["waveform"].shape), (2, 1, 12000))

    def test_stereo_node_converts_mono_and_keeps_batches(self):
        source = make_audio(batches=2)
        result = Music_StereoEnhance().enhance(source, 0.5)[0]
        self.assertEqual(tuple(result["waveform"].shape), (2, 2, 48000))

    def test_noise_remove_uses_package_relative_import(self):
        source = make_audio(seconds=0.1)
        result = Music_NoiseRemove().remove_noise(source, "Hiss Only", 0.0)[0]
        self.assertEqual(tuple(result["waveform"].shape), tuple(source["waveform"].shape))
        self.assertIsNot(result, source)

    def test_mixer_resamples_and_broadcasts_mono(self):
        first = make_audio(sample_rate=48000, seconds=1.0, channels=2)
        second = make_audio(sample_rate=24000, seconds=0.5, channels=1)
        result = Music_AudioMixer().mix(first, second)[0]
        self.assertEqual(result["sample_rate"], 48000)
        self.assertEqual(tuple(result["waveform"].shape), (1, 2, 48000))


class DspRegressionTests(unittest.TestCase):
    def test_lufs_measurement_and_normalization(self):
        source = make_audio(seconds=2.0, amplitude=0.1)["waveform"].numpy()
        measured = calculate_lufs(source, 48000)
        self.assertAlmostEqual(measured, -23.05, delta=0.35)
        normalized = normalize_to_lufs(source, -14.0, 48000)
        self.assertAlmostEqual(calculate_lufs(normalized, 48000), -14.0, delta=0.2)

    def test_eq_applies_requested_gain_once(self):
        source = make_audio(seconds=2.0, amplitude=0.05, frequency=1000.0)["waveform"].numpy()
        processed = apply_eq(source, [1000.0], [6.0], 48000)
        start = 4800
        input_rms = np.sqrt(np.mean(source[..., start:] ** 2))
        output_rms = np.sqrt(np.mean(processed[..., start:] ** 2))
        gain_db = 20.0 * np.log10(output_rms / input_rms)
        self.assertAlmostEqual(gain_db, 6.0, delta=0.2)

    def test_gain_is_exact_and_not_peak_normalized(self):
        source = np.array([0.25, -0.5], dtype=np.float32)
        np.testing.assert_allclose(apply_gain(source, 6.0206), source * 2.0, rtol=2e-5)

    def test_batch_items_are_normalized_independently(self):
        first = np.array([[[0.6]], [[0.05]]], dtype=np.float32)
        mixed = mix_audio(first, first)
        np.testing.assert_allclose(np.max(np.abs(mixed), axis=(1, 2)), [1.0, 0.1])

        zeros = np.zeros_like(first)
        recombined = recombine_stems({"vocals": first * 2.0, "music": zeros})
        np.testing.assert_allclose(
            np.max(np.abs(recombined), axis=(1, 2)), [0.98, 0.1], rtol=1e-6
        )

    def test_multiband_compressor_uses_attack_and_links_stereo(self):
        sample_rate = 48000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        left = (0.05 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32)
        start = sample_rate // 2
        left[start : start + sample_rate // 10] = 0.9
        stereo = np.stack([left, left * 0.5])
        fast = apply_multiband_compression(
            stereo, sample_rate, threshold=0.1, ratio=6.0, attack_ms=1.0, release_ms=80.0
        )
        slow = apply_multiband_compression(
            stereo, sample_rate, threshold=0.1, ratio=6.0, attack_ms=50.0, release_ms=80.0
        )
        probe = start + int(0.005 * sample_rate)
        self.assertGreater(abs(float(slow[0, probe])), abs(float(fast[0, probe])))
        active = np.abs(fast[0]) > 1e-5
        np.testing.assert_allclose(fast[1, active] / fast[0, active], 0.5, atol=2e-5)

    def test_stems_are_distinct_and_recombine_at_unity(self):
        source = make_audio(seconds=0.2, amplitude=0.1)["waveform"].numpy()
        stems = separate_all_stems(source, 48000)
        self.assertFalse(np.array_equal(stems["vocals"], stems["music"]))
        recombined = recombine_stems(stems)
        np.testing.assert_allclose(recombined, source, atol=1e-6)

    def test_short_stems_do_not_fail(self):
        source = make_audio(seconds=0.01)["waveform"].numpy()
        stems = separate_all_stems(source, 48000)
        recombined = recombine_stems(stems)
        np.testing.assert_array_equal(recombined, source)

    def test_true_peak_limiter_is_transparent_below_ceiling(self):
        source = np.full((2, 10000), 0.5, dtype=np.float32)
        processed, reduction, _ = apply_true_peak_limiter(source, 48000, -1.0)
        np.testing.assert_array_equal(processed, source)
        self.assertEqual(reduction, 0.0)

    def test_true_peak_limiter_respects_ceiling(self):
        time = np.arange(48000, dtype=np.float32) / 48000
        source = (1.2 * np.sin(2 * np.pi * 19001 * time))[np.newaxis, :]
        processed, reduction, true_peak = apply_true_peak_limiter(source, 48000, -1.0)
        self.assertGreater(reduction, 0.0)
        self.assertLessEqual(true_peak, -0.99)
        self.assertLessEqual(measure_true_peak_db(processed, 4), -0.99)


class NewNodeTests(unittest.TestCase):
    def test_audio_repair_bypass_is_exact(self):
        source = make_audio()
        result, report = Music_AudioRepair().repair(source, "Off", 0.5, -1.0, 0.5, 10.0, -1.0)
        self.assertTrue(torch.equal(result["waveform"], source["waveform"]))
        self.assertIn("clicks=0", report)

    def test_audio_repair_removes_dc_and_click(self):
        sample_rate = 48000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        clean = 0.4 * np.sin(2 * np.pi * 997 * time)
        damaged = clean + 0.05
        damaged[10000] = 1.0
        repaired, report = repair_audio(
            damaged,
            sample_rate,
            mode="Auto (All)",
            sensitivity=0.7,
            clip_threshold_dbfs=-1.0,
        )
        self.assertLess(abs(float(np.mean(repaired))), 1e-4)
        self.assertLess(abs(float(repaired[10000] - clean[10000])), 0.02)
        self.assertIn("clicks=1", report)

    def test_audio_repair_node_repairs_nonfinite_samples(self):
        source = make_audio(seconds=0.1)
        source["waveform"][0, 0, 10] = float("nan")
        source["waveform"][0, 0, 20] = float("inf")
        result, report = Music_AudioRepair().repair(
            source, "Auto (All)", 0.5, -1.0, 0.5, 10.0, -1.0
        )
        self.assertTrue(torch.isfinite(result["waveform"]).all())
        self.assertIn("nonfinite=2", report)

    def test_audio_repair_improves_clipped_sine(self):
        sample_rate = 48000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        clean = 0.5 * np.sin(2 * np.pi * 997 * time)
        clipped = np.clip(clean, -0.3, 0.3)
        repaired, report = repair_audio(
            clipped,
            sample_rate,
            mode="De-clip Only",
            clip_threshold_dbfs=-12.0,
        )
        self.assertLess(np.mean((repaired - clean) ** 2), np.mean((clipped - clean) ** 2) * 0.2)
        self.assertIn("clip_events=", report)

    def test_audio_repair_does_not_modify_clean_loud_tones(self):
        sample_rate = 48000
        time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
        for frequency in (50.0, 440.0, 997.0, 12000.0):
            with self.subTest(frequency=frequency):
                clean = (0.95 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
                repaired, report = repair_audio(clean, sample_rate, mode="Auto (All)")
                np.testing.assert_array_equal(repaired, clean)
                self.assertIn("clip_events=0", report)
                self.assertIn("clicks=0", report)

    def test_audio_repair_ceiling_is_independent_per_batch(self):
        sample_rate = 48000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        hot = 0.95 * np.sin(2 * np.pi * 440.0 * time) + 0.05
        quiet = 0.10 * np.sin(2 * np.pi * 440.0 * time) + 0.05
        source = np.stack([hot, quiet]).astype(np.float32)[:, np.newaxis, :]
        repaired, _ = repair_audio(source, sample_rate, mode="Auto (All)")
        peaks = np.max(np.abs(repaired), axis=(1, 2))
        self.assertAlmostEqual(float(peaks[0]), 10.0 ** (-1.0 / 20.0), places=5)
        self.assertAlmostEqual(float(peaks[1]), 0.10, places=4)

    def test_every_genre_maps_to_a_valid_profile(self):
        self.assertEqual(len(GENRE_OPTIONS), len(set(GENRE_OPTIONS)))
        self.assertEqual(len(GENRE_OPTIONS), 200)
        for genre in GENRE_OPTIONS:
            self.assertIn(GENRE_TO_PROFILE[genre], PROFILES)

    def test_music_fix_compressor_honors_attack_time(self):
        sample_rate = 48000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        tone = (0.1 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32)
        transient_start = sample_rate // 2
        tone[transient_start : transient_start + sample_rate // 10] = 0.9
        stereo = np.stack([tone, tone])
        base = replace(PROFILES["club"], threshold_over_rms_db=3.0, max_gr_db=6.0)
        fast = _adaptive_compress(stereo, sample_rate, replace(base, attack_ms=1.0))
        slow = _adaptive_compress(stereo, sample_rate, replace(base, attack_ms=50.0))
        probe = transient_start + int(0.005 * sample_rate)
        self.assertGreater(slow[0, probe], fast[0, probe] + 0.2)

    def test_music_fix_preserves_shape_and_profile_ceiling(self):
        source = make_audio(seconds=2.0, channels=2, batches=2, amplitude=0.15)
        genre = "Pop / Mainstream Pop"
        result = Music_Fix().fix(source, genre)[0]
        self.assertEqual(tuple(result["waveform"].shape), tuple(source["waveform"].shape))
        self.assertEqual(result["sample_rate"], source["sample_rate"])
        output = result["waveform"].numpy()
        self.assertTrue(np.isfinite(output).all())
        ceiling = PROFILES[GENRE_TO_PROFILE[genre]].ceiling_dbtp
        self.assertLessEqual(measure_true_peak_db(output, 4), ceiling + 0.02)


if __name__ == "__main__":
    unittest.main()
