# ComfyUI Music Tools

Audio repair, genre-informed finishing, loudness, EQ, dynamics, stereo, and utility nodes for ComfyUI.

The pack uses ComfyUI's standard `AUDIO` value: a dictionary containing a float waveform shaped `[batch, channels, samples]` and an integer sample rate. Mono, stereo, and multi-item batches are preserved.

## Nodes

| Node | Purpose |
|---|---|
| **Music Fix** | Applies a conservative finishing profile chosen from 200 genre and subgenre labels. Profiles combine subtle EQ, linked compression, stereo shaping, LUFS management, and peak protection. |
| **Music - Audio Repair** | Repairs invalid samples, DC offset, isolated clicks/pops, and short hard-clipping plateaus. Also returns a repair report. |
| **Music - Master Audio Enhancement** | Configurable mastering chain with denoise, EQ, multiband dynamics, clarity, optional vocal processing, stereo shaping, loudness, and true-peak protection. |
| **Music - Noise Remove** | High-frequency hiss reduction or full stationary denoise. |
| **Music - Audio Upscale** | Band-limited sample-rate conversion from 16 kHz to 192 kHz. |
| **Music - Stereo Enhance** | Mid/side widening; mono input is safely duplicated to stereo. |
| **Music - LUFS Normalizer** | Integrated loudness measurement and normalization using ITU-R BS.1770 through `pyloudnorm`. |
| **Music - Equalize** | Nyquist-safe three-band peaking EQ. |
| **Music - Compressor** | Stereo-linked dynamic range compression. |
| **Music - Reverb** | Lightweight multi-tap ambience. |
| **Music - Gain** | Exact gain adjustment in dB. |
| **Music - Audio Mixer** | Mixes two tracks with sample-rate conversion, channel matching, and duration padding. |
| **Music - Audio Trimmer** | Extracts a validated time range. |
| **Music - Stem Separation** | Fast heuristic DSP split into vocals, drums, bass, music, and residual. |
| **Music - Stem Recombination** | Recombines five stems with individual levels and safe headroom. |

## Music Fix

`Music Fix` has exactly two inputs:

- `audio`: the input `AUDIO`;
- `genre`: a dropdown covering broad families and subgenres across pop, rock, metal, hip-hop, electronic, dance, classical, jazz, soul, country, Latin, Brazilian, African, reggae, ambient, lo-fi, soundtrack, folk, world music, and experimental music.

The dropdown maps related subgenres to 18 maintained DSP profiles. The profile is a starting point, not an automatic genre detector or a guarantee of a particular artistic result. EQ changes are deliberately subtle, stereo compression is linked, bass widening is avoided, and the peak ceiling takes priority when the requested LUFS target is not safely reachable.

Example:

```text
Load Audio -> Music Fix (genre: Brazilian / MPB) -> Save Audio
```

## Audio Repair

Place repair before denoise or mastering so impulsive defects are not spread by spectral processing:

```text
Load Audio -> Audio Repair -> Noise Remove -> Music Fix -> Save Audio
```

Modes:

- `Auto (All)`: invalid samples, meaningful DC offset, short clipping plateaus, and isolated clicks;
- `De-click Only`;
- `De-clip Only`;
- `DC Offset Only`;
- `Off`: bit-exact bypass after format conversion.

The repair is conservative. Long flat regions are skipped because they may be intentional synthesis or distortion.

## Installation

Clone into `ComfyUI/custom_nodes` and install the core dependencies with ComfyUI's Python:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jeankassio/ComfyUI_MusicTools.git
cd ComfyUI_MusicTools
python -m pip install -r requirements.txt
```

For the Windows portable build:

```bat
cd ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/jeankassio/ComfyUI_MusicTools.git
cd ComfyUI_MusicTools
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

Restart ComfyUI. The nodes appear in the `music` category.

### Optional MetricGAN+ speech enhancement

The `ai_enhance` option in Master Audio Enhancement is optional and intended for speech/vocal material, not transparent full-mix mastering. Install its extra packages only if needed:

```bash
python -m pip install -r requirements-ai.txt
```

Use the `torchaudio` build already matched to ComfyUI's installed `torch`; do not independently upgrade or downgrade those packages. If the optional stack is unavailable or incompatible, the master node continues with its DSP chain.

## Practical loudness targets

Common starting points, not universal delivery rules:

- streaming music: around `-14 LUFS`;
- spoken-word/podcast: commonly around `-16 LUFS` stereo;
- EBU R128 programme loudness: `-23 LUFS`;
- louder modern masters may use lower numerical targets, but dynamics and true-peak headroom should be evaluated rather than assumed.

LUFS normalization may finish below its target when peak headroom is insufficient. Use loudness and true-peak measurements together.

## Important limitations

- Sample-rate conversion does not recreate frequencies or detail missing from the source.
- The stem node is a fast harmonic/percussive/frequency heuristic, not Demucs or another neural source-separation model.
- Processing cannot guarantee “studio quality”; source quality and artistic decisions still matter.
- The vocal naturalizer is a deterministic artifact smoother. It does not reconstruct pitch or undo pitch correction.

## Development and tests

The regression suite uses the Python standard library:

```bash
python -m unittest discover -s tests -v
```

It covers ComfyUI batch preservation, sample-rate metadata, LUFS, EQ gain, mixing, stems, true-peak limiting, Audio Repair, Music Fix mappings, and profile ceilings.

## License

MIT. See [LICENSE](LICENSE).

Current version: **1.1.0**.
