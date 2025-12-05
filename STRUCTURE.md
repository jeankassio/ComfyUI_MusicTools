# 📦 ComfyUI Music Tools - Project Structure

**Organized on**: December 5, 2025  
**Built with**: GitHub Copilot

---

## 🎯 Clean Root Directory

```
ComfyUI_MusicTools/
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
├── README.md               # Main documentation (public)
├── requirements.txt        # Python dependencies
├── nodes.py                # ComfyUI node definitions
├── __init__.py             # Package entry point
├── pyrightconfig.json      # Type checker config
│
├── 📁 src/                 # Core audio processing modules
├── 📁 tests/               # Unit and integration tests
├── 📁 scripts/             # Development utilities
└── 📁 docs/                # Internal documentation
```

---

## 📂 Folder Contents

### `src/` - Core Modules (6 files)
```
src/
├── __init__.py                  # Package exports
├── config.py                    # Configuration settings
├── utils.py                     # Audio utilities (LUFS, EQ, compression, etc.)
├── vocal_enhance.py             # Vocal processing (de-esser, naturalizer)
├── enhanced_master_audio.py     # Main processing pipeline
├── master_audio.py              # Original master audio implementation
└── stereo_enhance.py            # Stereo widening
```

**Purpose**: All audio processing logic isolated from ComfyUI interface.

---

### `tests/` - Test Suite (17 files)
```
tests/
├── test_vocal_naturalizer.py   # Vocal naturalizer tests
├── test_enhanced_master.py     # Main pipeline tests
├── test_limiter_speed.py       # Performance benchmarks
├── test_vocal_enhance_speed.py # Vocal processing benchmarks
├── test_comprehensive.py       # Integration tests
├── test_integration.py         # End-to-end tests
├── test_master_audio.py        # Master audio tests
├── test_nodes.py               # ComfyUI node tests
└── ... (9 more test files)
```

**Purpose**: Validate functionality and performance.

---

### `scripts/` - Development Tools (9 files)
```
scripts/
├── install_dependencies.py     # Auto-install requirements
├── quick_start.py              # Quick start examples
├── examples.py                 # Usage examples
├── print_final_summary.py      # Project summary
└── ... (5 optimization/changelog scripts)
```

**Purpose**: Helper scripts for development and user convenience.

---

### `docs/` - Documentation (35 files)
```
docs/
├── PROJECT_STRUCTURE.md        # This file
├── VOCAL_NATURALIZER.md        # Vocal naturalizer documentation
├── OPTIMIZATION_NOTES.md       # Performance optimization notes
├── PERFORMANCE.md              # Performance benchmarks
├── LIMITER_OPTIMIZATION.md     # Limiter optimization details
├── ARCHITECTURE.md             # System architecture
└── ... (29 more internal docs)
```

**Purpose**: Internal development documentation (ignored by git).

---

## 🔧 Import Structure

### From ComfyUI (nodes.py)
```python
from .src.utils import audio_to_numpy, calculate_lufs, ...
from .src.enhanced_master_audio import process_audio_stems
```

### Inside src/ modules
```python
from .vocal_enhance import apply_vocal_naturalizer
from .utils import calculate_lufs
```

### From external code
```python
from ComfyUI_MusicTools.nodes import Music_MasterAudioEnhancement
from ComfyUI_MusicTools.src.vocal_enhance import apply_vocal_naturalizer
```

---

## ✅ Benefits of This Structure

1. **Clean Root**: Only 7 essential files in root directory
2. **Separation of Concerns**: UI (nodes.py) separated from logic (src/)
3. **Easy Testing**: All tests organized in dedicated folder
4. **Professional**: GitHub-ready with proper Python package structure
5. **Maintainable**: Clear module responsibilities
6. **Hidden Internals**: Development docs and scripts hidden from users

---

## 🚀 Quick Navigation

| Need to...                     | Go to...                  |
|-------------------------------|---------------------------|
| Add audio processing feature  | `src/utils.py` or `src/vocal_enhance.py` |
| Modify ComfyUI interface      | `nodes.py`                |
| Add tests                     | `tests/test_*.py`         |
| Create utility script         | `scripts/`                |
| Write documentation           | `docs/`                   |
| Update public docs            | `README.md` (root)        |

---

## 📝 Git Tracking

**Tracked** (visible on GitHub):
- Root files (README, LICENSE, requirements, etc.)
- `src/` modules
- `nodes.py` and `__init__.py`

**Ignored** (hidden from GitHub):
- `docs/*.md` (except this file)
- `tests/test_*.py`
- `scripts/*.py`
- `__pycache__/`
- `.vscode/`

See `.gitignore` for complete rules.

---

## 🎉 Summary

Before:
```
❌ 60+ files in root directory
❌ Test files mixed with source
❌ Documentation scattered
❌ Hard to navigate
```

After:
```
✅ 7 files in root directory
✅ Organized into 4 logical folders
✅ Clean separation of concerns
✅ Professional GitHub structure
✅ Easy to maintain and extend
```

---

**Made with ❤️ and GitHub Copilot**
