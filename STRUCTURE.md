# Project structure

```text
ComfyUI_MusicTools/
├── __init__.py                 # ComfyUI node registration
├── nodes.py                    # Legacy/V1 ComfyUI node interfaces
├── node_list.json              # Manager/Registry node catalog
├── pyproject.toml              # Registry metadata
├── requirements.txt            # Core dependencies
├── requirements-ai.txt         # Optional MetricGAN+ dependencies
├── src/
│   ├── audio_repair.py         # De-click, de-clip, DC and invalid-sample repair
│   ├── genre_presets.py        # Music Fix taxonomy and DSP profiles
│   ├── limiter.py              # Stereo-linked true-peak limiter
│   ├── utils.py                # Format conversion and general DSP
│   ├── master_audio.py         # Mastering stages
│   ├── enhanced_master_audio.py# Master pipeline and optional AI enhancement
│   ├── stereo_enhance.py       # Stereo imaging
│   ├── vocal_enhance.py        # Vocal-oriented artifact processing
│   └── config.py               # Public defaults/reference values
└── tests/
    └── test_audio_tools.py     # Unit and regression suite
```

Run validation from the repository root:

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

ComfyUI audio is represented as `[batch, channels, samples]` internally. DSP helpers in `src/` preserve that layout unless their documentation explicitly accepts an unbatched `[channels, samples]` or mono `[samples]` array.
