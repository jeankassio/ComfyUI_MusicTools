# Installation Instructions

This custom node pack is designed to be installed via ComfyUI Manager or manually.

## Via ComfyUI Manager (Recommended)

1. Open ComfyUI Manager in ComfyUI
2. Click "Install Custom Nodes"
3. Search for "ComfyUI Music Tools"
4. Click Install
5. Restart ComfyUI

## Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jeankassio/ComfyUI_MusicTools.git
cd ComfyUI_MusicTools
pip install -r requirements.txt
```

## Windows Portable Installation

```bash
cd ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/jeankassio/ComfyUI_MusicTools.git
cd ComfyUI_MusicTools
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

## Dependencies

All dependencies are listed in `requirements.txt` and will be automatically installed:

- numpy>=1.21.0
- scipy>=1.7.0
- librosa>=0.9.0
- soundfile>=0.10.0
- pyloudnorm>=0.1.0
- noisereduce>=2.0.0
- torch>=1.9.0
- torchaudio>=0.9.0

Optional (for AI enhancement):
- speechbrain (auto-installs when needed)
- huggingface-hub (auto-installs when needed)

Optional (for stem separation):
- spleeter or demucs (install manually if needed)

## Verification

After installation, restart ComfyUI and verify that nodes starting with "Music -" appear in the node menu under the "audio" category.

## Troubleshooting

If nodes don't appear:
1. Check that all dependencies installed correctly
2. Restart ComfyUI completely
3. Check ComfyUI console for error messages
4. Verify Python version is 3.8 or higher

For more help, see the main README.md or open an issue on GitHub.
