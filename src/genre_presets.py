"""Conservative genre-informed finishing presets for the Music Fix node."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .limiter import measure_true_peak_db
from .utils import apply_eq, calculate_lufs


@dataclass(frozen=True)
class GenreProfile:
    low_db: float
    mid_db: float
    high_db: float
    threshold_over_rms_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    max_gr_db: float
    width: float
    target_lufs: float
    ceiling_dbtp: float


PROFILES = {
    "balanced": GenreProfile(0.0, 0.0, 0.0, 9.0, 1.35, 30, 180, 1.2, 1.00, -14.0, -1.0),
    "classical": GenreProfile(0.0, 0.0, 0.2, 12.0, 1.15, 50, 350, 0.6, 1.00, -18.0, -1.5),
    "cinematic": GenreProfile(0.2, 0.1, 0.3, 11.0, 1.25, 45, 300, 0.9, 1.06, -16.5, -1.5),
    "acoustic": GenreProfile(0.2, 0.3, 0.3, 10.0, 1.30, 35, 220, 1.0, 1.02, -16.0, -1.2),
    "jazz_soul": GenreProfile(0.2, 0.3, 0.2, 9.0, 1.35, 35, 200, 1.2, 1.02, -15.5, -1.2),
    "vocal_modern": GenreProfile(0.1, 0.5, 0.3, 8.0, 1.50, 20, 140, 1.6, 1.03, -14.0, -1.0),
    "pop": GenreProfile(0.4, 0.3, 0.5, 7.0, 1.60, 15, 110, 1.8, 1.05, -12.5, -1.0),
    "rock": GenreProfile(0.3, 0.5, 0.2, 7.0, 1.65, 22, 130, 1.8, 1.03, -13.0, -1.0),
    "heavy": GenreProfile(0.4, 0.6, -0.2, 6.5, 1.75, 15, 100, 2.0, 1.02, -12.5, -1.0),
    "hiphop": GenreProfile(0.6, -0.2, 0.2, 7.0, 1.60, 25, 150, 1.8, 1.04, -12.5, -1.0),
    "bass_electronic": GenreProfile(0.7, -0.2, 0.4, 6.0, 1.75, 20, 110, 2.2, 1.06, -12.0, -1.0),
    "club": GenreProfile(0.6, 0.0, 0.5, 5.5, 1.85, 12, 85, 2.5, 1.08, -11.5, -1.0),
    "fast_electronic": GenreProfile(0.5, 0.1, 0.5, 6.0, 1.85, 8, 65, 2.3, 1.07, -12.0, -1.0),
    "ambient": GenreProfile(0.1, -0.1, 0.4, 12.0, 1.15, 60, 400, 0.7, 1.10, -17.0, -1.5),
    "lofi": GenreProfile(0.3, 0.3, -0.8, 9.0, 1.35, 30, 220, 1.1, 1.02, -15.0, -1.5),
    "reggae_dub": GenreProfile(0.7, -0.3, -0.2, 8.0, 1.45, 30, 180, 1.4, 1.04, -13.5, -1.2),
    "latin_acoustic": GenreProfile(0.3, 0.4, 0.3, 9.0, 1.45, 25, 160, 1.4, 1.03, -14.5, -1.2),
    "latin_dance": GenreProfile(0.6, 0.2, 0.4, 6.0, 1.75, 14, 95, 2.1, 1.06, -12.0, -1.0),
}


GENRE_GROUPS = {
    "balanced": (
        "General / Balanced", "General / Other or Unlisted", "Experimental / Avant-Garde",
        "Experimental / Noise", "Experimental / Musique Concrète", "Children / Children's Music",
    ),
    "classical": (
        "Classical / Classical", "Classical / Medieval", "Classical / Renaissance",
        "Classical / Baroque", "Classical / Romantic", "Classical / Contemporary Classical",
        "Classical / Chamber Music", "Classical / Orchestral", "Classical / Opera or Choral",
        "Classical / Solo Piano", "Classical / Minimalism", "Indian / Hindustani Classical",
        "Indian / Carnatic Classical",
    ),
    "cinematic": (
        "Soundtrack / Film Score", "Soundtrack / Cinematic", "Soundtrack / Game Music",
        "Soundtrack / Trailer Music", "Soundtrack / Musical Theatre", "Soundtrack / Anime Score",
    ),
    "acoustic": (
        "Acoustic / Acoustic", "Acoustic / Folk", "Acoustic / Singer-Songwriter",
        "Country / Country", "Country / Americana", "Country / Bluegrass", "Country / Outlaw Country",
        "World / Traditional", "World / Celtic", "World / Fado",
        "World / Middle Eastern", "World / Arabic", "Christian / Worship",
    ),
    "jazz_soul": (
        "Jazz / Jazz", "Jazz / Bebop", "Jazz / Cool Jazz", "Jazz / Smooth Jazz", "Jazz / Big Band",
        "Jazz / Fusion", "Jazz / Latin Jazz", "Blues / Blues", "Blues / Delta Blues",
        "Soul / Soul", "Soul / Neo Soul", "Soul / Funk", "Soul / Motown", "Soul / Gospel",
    ),
    "vocal_modern": (
        "R&B / Contemporary R&B", "R&B / Alternative R&B", "R&B / Quiet Storm",
        "Vocal / A Cappella", "Vocal / Spoken Word",
    ),
    "pop": (
        "Pop / Mainstream Pop", "Pop / Indie Pop", "Pop / Electropop", "Pop / Synthpop",
        "Pop / Dream Pop", "Pop / Art Pop", "Pop / K-Pop", "Pop / J-Pop",
        "Pop / City Pop", "Pop / C-Pop", "Pop / Cantopop", "Pop / Mandopop", "Pop / Latin Pop",
        "Indian / Bollywood", "Country / Country Pop",
    ),
    "rock": (
        "Rock / Rock", "Rock / Classic Rock", "Rock / Alternative Rock", "Rock / Indie Rock",
        "Rock / Progressive Rock", "Rock / Psychedelic Rock", "Rock / Garage Rock", "Rock / Grunge",
        "Rock / Punk Rock", "Rock / Pop Punk", "Rock / Post-Punk", "Rock / New Wave",
        "Rock / Shoegaze", "Rock / Emo", "Rock / Post-Rock", "Rock / Ska", "Rock / Surf Rock",
        "Rock / Southern Rock",
    ),
    "heavy": (
        "Heavy / Hard Rock", "Heavy / Heavy Metal", "Heavy / Thrash Metal", "Heavy / Death Metal",
        "Heavy / Black Metal", "Heavy / Doom Metal", "Heavy / Power Metal", "Heavy / Symphonic Metal",
        "Heavy / Metalcore", "Heavy / Hardcore", "Heavy / Djent", "Heavy / Nu Metal",
        "Heavy / Industrial Metal", "Heavy / Grindcore",
    ),
    "hiphop": (
        "Hip-Hop / Hip-Hop", "Hip-Hop / Rap", "Hip-Hop / Boom Bap", "Hip-Hop / Conscious Hip-Hop",
        "Hip-Hop / Alternative Hip-Hop", "Hip-Hop / Drill", "Hip-Hop / Grime", "Hip-Hop / Brazilian Rap",
    ),
    "bass_electronic": (
        "Hip-Hop / Trap", "Hip-Hop / Phonk", "Hip-Hop / Brazilian Trap", "Pop / Hyperpop", "Electronic / Dubstep",
        "Electronic / Future Bass", "Electronic / Bass Music", "Electronic / Synthwave",
        "Electronic / Witch House",
    ),
    "club": (
        "Dance / Dance Pop", "Dance / EDM", "Dance / House", "Dance / Deep House",
        "Dance / Tech House", "Dance / Progressive House", "Dance / Acid House", "Dance / Electro House",
        "Dance / Techno", "Dance / Minimal Techno", "Dance / Detroit Techno", "Dance / Trance",
        "Dance / Psytrance", "Dance / Disco", "Dance / Nu Disco", "Dance / Eurodance",
        "Dance / UK Garage", "Dance / Amapiano", "African / Gqom", "African / Kuduro",
    ),
    "fast_electronic": (
        "Electronic / Drum & Bass", "Electronic / Jungle", "Electronic / Breakbeat",
        "Electronic / Hardstyle", "Electronic / Hardcore or Gabber", "Electronic / Footwork",
    ),
    "ambient": (
        "Electronic / Ambient", "Electronic / Downtempo", "Electronic / Chillout", "Electronic / New Age",
        "Electronic / Drone", "Electronic / IDM", "Electronic / Dark Ambient", "Electronic / Space Music",
    ),
    "lofi": (
        "Lo-Fi / Lo-Fi", "Lo-Fi / Lo-Fi Hip-Hop", "Lo-Fi / Chillhop", "Lo-Fi / Trip-Hop",
        "Lo-Fi / Vaporwave", "Lo-Fi / Bedroom Pop", "Lo-Fi / Slushwave",
    ),
    "reggae_dub": (
        "Reggae / Reggae", "Reggae / Roots Reggae", "Reggae / Dub", "Reggae / Dancehall",
        "Reggae / Rocksteady",
    ),
    "latin_acoustic": (
        "Brazilian / Bossa Nova", "Brazilian / Samba", "Brazilian / Pagode", "Brazilian / MPB",
        "Brazilian / Forró", "Brazilian / Sertanejo", "Brazilian / Choro",
        "Brazilian / Maracatu", "Latin / Flamenco",
        "Latin / Tango", "Latin / Bolero", "Latin / Mariachi", "Latin / Corrido",
    ),
    "latin_dance": (
        "Latin / Reggaeton", "Latin / Dembow", "Latin / Salsa", "Latin / Merengue",
        "Latin / Bachata", "Latin / Cumbia", "Latin / Latin Electronic", "Brazilian / Funk Carioca",
        "Brazilian / Funk Melody", "Brazilian / Brega Funk", "Brazilian / Tecnobrega",
        "Brazilian / Lambada", "Brazilian / Piseiro", "Brazilian / Axé", "Brazilian / Frevo",
        "African / Afrobeat", "African / Afrobeats", "African / Highlife", "African / Soukous",
        "African / Mbalax",
    ),
}

GENRE_TO_PROFILE = {
    genre: profile_name
    for profile_name, genres in GENRE_GROUPS.items()
    for genre in genres
}
GENRE_OPTIONS = tuple(GENRE_TO_PROFILE)


def _active_rms_db(batch: np.ndarray, sample_rate: int) -> float:
    block_size = max(1, int(round(0.4 * sample_rate)))
    power = np.mean(np.square(batch, dtype=np.float64), axis=0)
    block_values = []
    for start in range(0, power.size, block_size):
        block = power[start : start + block_size]
        if block.size:
            block_values.append(float(np.sqrt(np.mean(block))))
    if not block_values:
        return -120.0
    levels = 20.0 * np.log10(np.maximum(block_values, 1e-12))
    active = levels[levels > max(-70.0, float(np.max(levels)) - 40.0)]
    return float(np.median(active)) if active.size else -120.0


def _adaptive_compress(batch: np.ndarray, sample_rate: int, profile: GenreProfile) -> np.ndarray:
    if batch.shape[-1] == 0:
        return batch.copy()
    active_rms = _active_rms_db(batch, sample_rate)
    peak = float(np.max(np.abs(batch)))
    if peak <= 1e-8 or active_rms <= -100.0:
        return batch.copy()
    crest_db = 20.0 * np.log10(peak) - active_rms
    if crest_db < 5.0:
        return batch.copy()
    max_gr = profile.max_gr_db * (0.5 if crest_db < 8.0 else 1.0)
    threshold_db = active_rms + profile.threshold_over_rms_db

    detector = np.max(np.abs(batch), axis=0)
    attack_samples = max(1.0, profile.attack_ms * sample_rate / 1000.0)
    attack_alpha = float(np.exp(-1.0 / attack_samples))
    # Smooth the rising detector itself, so attack_ms controls how quickly gain
    # reduction engages instead of merely looking ahead by that duration.
    envelope, _ = signal.lfilter(
        [1.0 - attack_alpha],
        [1.0, -attack_alpha],
        detector,
        zi=[attack_alpha * detector[0]],
    )
    level_db = 20.0 * np.log10(np.maximum(envelope, 1e-12))
    desired_db = -np.maximum(level_db - threshold_db, 0.0) * (1.0 - 1.0 / profile.ratio)
    desired_db = np.maximum(desired_db, -max_gr)
    desired_gain = np.power(10.0, desired_db / 20.0).astype(np.float32)
    alpha = float(np.exp(-1.0 / max(1.0, profile.release_ms * sample_rate / 1000.0)))
    smoothed, _ = signal.lfilter(
        [1.0 - alpha], [1.0, -alpha], desired_gain, zi=[alpha * desired_gain[0]]
    )
    gain = np.minimum(smoothed, desired_gain).astype(np.float32)
    return (batch * gain[np.newaxis, :]).astype(np.float32)


def _shape_stereo(batch: np.ndarray, sample_rate: int, width: float) -> np.ndarray:
    if batch.shape[0] != 2 or batch.shape[-1] == 0 or abs(width - 1.0) < 1e-8:
        return batch.copy()
    left, right = batch
    if np.std(left) > 1e-10 and np.std(right) > 1e-10:
        correlation = float(np.corrcoef(left, right)[0, 1])
        safety = float(np.clip((correlation + 0.2) / 0.7, 0.0, 1.0))
        width = 1.0 + (width - 1.0) * safety
    else:
        width = 1.0
    if abs(width - 1.0) < 1e-8:
        return batch.copy()

    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    cutoff = min(180.0, sample_rate * 0.1)
    sos = signal.butter(2, cutoff, btype="highpass", fs=sample_rate, output="sos")
    side_high = signal.sosfilt(sos, side)
    side_low = side - side_high
    side_shaped = side_low + side_high * width
    return np.stack([mid + side_shaped, mid - side_shaped]).astype(np.float32)


def apply_genre_finish(audio: np.ndarray, sample_rate: int, genre: str) -> np.ndarray:
    """Apply the selected genre-informed profile without changing shape or duration."""
    if genre not in GENRE_TO_PROFILE:
        raise ValueError(f"Unknown genre preset: {genre}")
    profile = PROFILES[GENRE_TO_PROFILE[genre]]
    work = np.asarray(audio, dtype=np.float32)
    original_ndim = work.ndim
    if original_ndim == 1:
        work = work[np.newaxis, np.newaxis, :]
    elif original_ndim == 2:
        work = work[np.newaxis, :, :]
    elif original_ndim != 3:
        raise ValueError(f"Unsupported audio shape: {work.shape}")
    if not np.isfinite(work).all():
        raise ValueError("Audio contains NaN or infinite values")

    frequencies = [90.0, 2000.0, min(9000.0, sample_rate * 0.42)]
    work = apply_eq(
        work,
        frequencies,
        [profile.low_db, profile.mid_db, profile.high_db],
        sample_rate,
    )

    result = np.empty_like(work, dtype=np.float32)
    oversample = 4 if sample_rate <= 48000 else 2
    for index, batch in enumerate(work):
        shaped = _adaptive_compress(batch, sample_rate, profile)
        shaped = _shape_stereo(shaped, sample_rate, profile.width)
        current_lufs = calculate_lufs(shaped, sample_rate)
        if np.isfinite(current_lufs) and shaped.size:
            true_peak = measure_true_peak_db(shaped, oversample)
            desired_gain_db = profile.target_lufs - current_lufs
            ceiling_gain_db = profile.ceiling_dbtp - true_peak
            gain_db = min(desired_gain_db, ceiling_gain_db, 12.0)
            shaped = shaped * (10.0 ** (gain_db / 20.0))
        result[index] = shaped.astype(np.float32)

    if original_ndim == 1:
        return result[0, 0]
    if original_ndim == 2:
        return result[0]
    return result
