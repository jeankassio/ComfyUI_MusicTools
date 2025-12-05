"""
Configuration file for ComfyUI Music Tools
Customize default parameters and behavior here
"""

# Global Audio Settings
AUDIO_CONFIG = {
    # Default sample rate (will auto-detect from input in future versions)
    "default_sample_rate": 44100,
    
    # Maximum audio length in seconds to process at once
    "max_audio_length": 600,  # 10 minutes
    
    # Normalize output to prevent clipping
    "normalize_output": True,
    
    # Clipping threshold
    "clipping_threshold": 1.0,
}

# Noise Removal Settings
NOISE_REMOVAL_CONFIG = {
    "default_intensity": 0.5,
    
    # FFT settings for STFT analysis
    "frame_length": 2048,
    "hop_length": 512,
    
    # Noise profile estimation
    "noise_profile_duration": 0.1,  # First 10% for noise estimation
    
    # Over-subtraction prevention
    "over_subtraction_factor": 0.1,
}

# Audio Upscaling Settings
UPSCALE_CONFIG = {
    "default_target_sr": 48000,
    "default_restore_sr": False,
    
    # Supported sample rates
    "supported_sample_rates": [
        16000, 22050, 24000, 32000, 44100, 48000,
        88200, 96000, 176400, 192000
    ],
}

# Stereo Enhancement Settings
STEREO_CONFIG = {
    "default_intensity": 0.5,
    "min_intensity": 0.0,
    "max_intensity": 2.0,
}

# LUFS Normalization Settings
LUFS_CONFIG = {
    "default_target_lufs": -14.0,
    
    # LUFS standards
    "standards": {
        "spotify": -23,
        "youtube": -14,
        "broadcast": -14,  # EBU R128
        "podcast": -16,
        "gaming": -18,
    },
    
    # Headroom to prevent clipping
    "max_headroom_db": 1.0,
}

# EQ Settings
EQ_CONFIG = {
    "default_bass_freq": 100,
    "default_mid_freq": 1000,
    "default_treble_freq": 10000,
    
    "default_gains": {
        "bass": 0.0,
        "mid": 0.0,
        "treble": 0.0,
    },
    
    # EQ Q factor (bandwidth)
    "q_factor": 1.0,
}

# Reverb Settings
REVERB_CONFIG = {
    "default_decay": 0.5,
    
    # Delay times for reverb (in seconds)
    "delay_times": [0.029, 0.031, 0.037, 0.041],
    
    # Wet/dry mix
    "default_mix": 0.5,
}

# Compression Settings
COMPRESSION_CONFIG = {
    "default_threshold": 0.5,
    "default_ratio": 4.0,
    
    # Attack and release times (in ms)
    "attack_time": 5,
    "release_time": 100,
}

# Gain Settings
GAIN_CONFIG = {
    "default_gain_db": 0.0,
    "min_gain_db": -24.0,
    "max_gain_db": 24.0,
}

# Mixer Settings
MIXER_CONFIG = {
    # Gain stage for each input track
    "default_gain_db": -6.0,
    
    # Normalize mix output
    "normalize_output": True,
}

# Trimmer Settings
TRIMMER_CONFIG = {
    "default_start_time": 0.0,
    "default_end_time": 10.0,
}

# Processing Settings
PROCESSING_CONFIG = {
    # Use float32 for audio processing
    "data_type": "float32",
    
    # Use batch processing where possible
    "enable_batch_processing": True,
    
    # Number of worker threads
    "num_workers": 4,
}

# Feature Flags
FEATURES = {
    "enable_gpu_acceleration": False,  # Set to True if GPU available
    "enable_advanced_noise_removal": False,  # Future feature
    "enable_stem_separation": False,  # Future feature
    "enable_ai_processing": False,  # Future feature
}

# Debug Settings
DEBUG = {
    "verbose": False,
    "log_processing_time": False,
    "save_intermediate_results": False,
    "intermediate_results_path": "./debug/",
}
