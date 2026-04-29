"""Preprocess raw voice datasets into one standardized dataset.

The output is a shared metadata CSV plus WAV files in ``dataset/``. Signal
normalization metrics are currently finalized only for SVD and AVFAD.
"""

# ============================================================
# Imports
# ============================================================

import re
from pathlib import Path
from typing import Never, Literal
from pydantic import BaseModel, Field

import nspfile
import soundfile as sf
import librosa
import wfdb
from tqdm import tqdm
import numpy as np
import parselmouth
from parselmouth.praat import call
import pandas as pd

from voice_pipeline.feature_extraction import normalize_sex, f0_range_for_sex


# ============================================================
# Folder structure
# ============================================================

OUT_DIR = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)

OUT_CSV = OUT_DIR / "metadata.csv"

SVD_ROOT = Path("data", "SVD")
VOICED_ROOT = Path("data") / "VOICED"


# ============================================================
# Metadata, changes to metadata or drop flags
# ============================================================

to_drop = pd.read_csv("to_drop.csv")

svd_metadata = pd.read_csv(SVD_ROOT / "svd_information.csv")
svd_metadata_by_session = {int(row["sessionid"]): row for row in svd_metadata.to_dict("records")}


# ============================================================
# Target metadatta scheme (FIXED)
# ============================================================


class Metadata(BaseModel):
    id: int = Field(default=-1)  # unique record ID (assigned during processing)
    talkerid: str  # unique talker ID (in form {dataset}_{original_id} to avoid clashes across datasets)
    talkersex: Literal["m", "f"]  # normalized sex to m/f
    age: int  # age in years
    pathologies: str  # semicolon-separated list of pathologies or "healthy"
    file: str  # original source file name
    vowel: Literal["a", "e", "i", "o", "u"]  # vowel (eg. "a")
    intonation: str  # intonation (eg. "n" for neutral) # TODO: define possible values
    dataset: str  # dataset name (eg. "SVD")
    computable_f0: bool | None = Field(default=None)  # whether the mean F0 is computable for the signal


# ============================================================
# Filtering
# ============================================================

MIN_AGE = 18


def is_adult(age: int) -> bool:
    """Check whether a participant is an adult.

    Parameters
    ----------
    age : object
        Age value that can be converted to ``float``.

    Returns
    -------
    bool
        ``True`` if ``age`` is greater than or equal to ``MIN_AGE``.
    """
    return float(age) >= MIN_AGE


# ============================================================
# Shared signal preprocessing
# ============================================================

# TODO @Galtan-dev: add comments
TRIM_TOP_DB = 20    # relict, asi uplně vyhodit
# target loundness to which is signal normalized
TARGET_RMS = 0.1
# threshold under which is normalization not applied (protection against amplification of noise)
MIN_RMS = 1e-6

# AVFAD crop parametry
# gues of time of silence on the beginning of signal
NOISE_EST_S = 0.1
SILENCE_RATIO = 0.3
MIN_PAUSE_S = 0.15
MIN_REP_S = 0.5
WINDOW_S = 0.05


# TODO @Galtan-dev: why this?!?
def frame_energy_db(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """
    Function which computes energy of sound signal in decibels
    """
    # coalculate number of frames that fit into the signal
    n_frames = (len(x) - frame_length) // hop_length + 1
    # if signal is too short, return -100 which is practically equal to silence
    if n_frames <= 0:
        return np.array([-100.0], dtype=np.float32)
    # if signal is long enough continue
    return np.array(
        [
            # calculate energy of signal for evry frame in signal and return mean of them
            10 * np.log10(np.mean(x[i * hop_length : i * hop_length + frame_length] ** 2) + 1e-10)
            for i in range(n_frames)
        ],
        dtype=np.float32,
    )


# TODO @Galtan-dev: why this?!?
def subtract_noise_energy(edb: np.ndarray, fs: int, hop_length: int) -> np.ndarray:
    """
    Function which substract guesed noise energy from sound signal.
    It ensures that recordings from low quality datasets or dataset
    recorded in noise conditions are cleaned form noise and suitable for
    trim and other preprocesing steps
    """
    # calculate how much of beginning of signal is used to guessed noise
    # fs/hop_length converts seconds to number of frames
    # min from len(edb) ensure that lenght of signal is not exceeded
    n_noise = min(max(1, int(NOISE_EST_S * fs / hop_length)), len(edb))
    # from selected frames is taken median of energy
    noise_level_db = float(np.median(edb[:n_noise]))
    # meadian of energy is substracted from original signal
    # max ensure that any value will not drop under 100 dB, which is considered to by silence
    return np.maximum(edb - noise_level_db, -100.0)


# TODO @Galtan-dev: why this?!?
def adaptive_silence_threshold(edb: np.ndarray) -> float:
    """
    Function for calcualtion of decibel level under which is all considered to by silence.
    Adaptive for various recordings.
    """
    # takes 10.percentil of energy of frames
    noise_floor = float(np.percentile(edb, 10))
    # takes 90.percentil of energy of frames
    voiced_peak = float(np.percentile(edb, 90))
    # now dynamical range of signal is taken and by silence ration
    # modifier level, under which is all considered to be noise, is set.
    # Level is set as SILENCE_RATIO percent from dynamical range
    return noise_floor + SILENCE_RATIO * (voiced_peak - noise_floor)


def find_first_repetition(x: np.ndarray, fs: int) -> tuple[int, list[tuple[int, int]], float]:
    """
    Function for cutting out first repetition of vowel from recordings in AVFAD dataset
    """
    # length of window in which search is conducted
    # minimaly length of 4
    frame_length = max(4, int(WINDOW_S * fs))
    # step of shift of window
    hop_length = max(1, frame_length // 2)
    # calculates frame energy for every window
    edb = frame_energy_db(x, frame_length, hop_length)
    # substract energy of noise from signal
    edb_clean = subtract_noise_energy(edb, fs, hop_length)
    # find adaptive threshold of noise
    thresh = adaptive_silence_threshold(edb_clean)
    # make bolean field, where signal is silence this is True
    silent = edb_clean < thresh
    n_frames = len(edb_clean)
    # return index of first frame which is not silenced
    first_voiced = next((i for i, s in enumerate(silent) if not s), None)
    # if nothing is found return length of signal and threshold
    if first_voiced is None:
        return len(x), [], thresh

    # definition of varibles
    pauses, in_pause, p_start = [], False, 0
    # cycle which walk through signal from first fram where is not silence
    for i in range(first_voiced, n_frames):
        # this if detect frame where is silence but algorithm is not yet in pause betwene repetitions
        # it is transition from speaking to silence
        if silent[i] and not in_pause:
            # index and flag is saved, from here algortim is "waiting" to silence end
            p_start, in_pause = i, True
        # this detects frame which is not silenced but algorithm is in pause
        elif not silent[i] and in_pause:
            # this checks minimal length of silence, short silence are not pauses
            # betwene repetitions but some patologies or mistakes
            if (i - p_start) * hop_length / fs >= MIN_PAUSE_S:
                # pause is long enough, save as tuple
                pauses.append((p_start * hop_length, i * hop_length))
            # resets flag, algorithm is back in voice
            in_pause = False
    # take care of posibility when transition form silence to voice ius not detected
    # minimal length is checked
    if in_pause and (n_frames - p_start) * hop_length / fs >= MIN_PAUSE_S:
        # save end of pause o
        pauses.append((p_start * hop_length, len(x)))

    rep_end = len(x)
    for pause_start, _ in pauses:
        if pause_start / fs >= MIN_REP_S:
            rep_end = pause_start
            break
    return rep_end, pauses, thresh


# TODO @Galtan-dev: why this?!?
def trim_edges(x: np.ndarray, fs: int) -> np.ndarray:
    fl = max(1, int(0.02 * fs))
    hl = max(1, fl // 2)
    edb = frame_energy_db(x, fl, hl)
    noise_floor = float(np.percentile(edb, 10))
    peak = float(np.percentile(edb, 95))
    start_thr = noise_floor + 0.25 * (peak - noise_floor)
    end_thr = noise_floor + 0.18 * (peak - noise_floor)
    min_run = max(1, int(0.06 * fs / hl))

    def first_run(mask, n):
        cnt = 0
        for i, v in enumerate(mask):
            cnt = cnt + 1 if v else 0
            if cnt >= n:
                return i - n + 1
        return None

    def last_run(mask, n):
        cnt = 0
        for i in range(len(mask) - 1, -1, -1):
            cnt = cnt + 1 if mask[i] else 0
            if cnt >= n:
                return i + n - 1
        return None

    s = first_run(edb >= start_thr, min_run)
    e = last_run(edb >= end_thr, min_run)
    if s is None or e is None:
        return x
    start, end = s * hl, min(len(x), e * hl + fl)
    return x[start:end] if end > start else x


# TODO @Galtan-dev: why this?!?
def rms_normalize(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x**2)))
    return (TARGET_RMS / rms) * x if rms > MIN_RMS else x


def preprocess_signal(x: np.ndarray, fs: int) -> np.ndarray:
    """Preprocess a sustained-vowel signal.

    The sequence is mono conversion, DC removal, edge trimming, and RMS normalization. It is used for SVD and VOICED.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray
        Preprocessed mono waveform as ``float32``.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, 0]
    x = x - np.mean(x)
    x = trim_edges(x, fs)
    x = rms_normalize(x)
    return x.astype(np.float32)


def preprocess_avfad_signal(x: np.ndarray, fs: int, crop_to_one_rep: bool = True) -> np.ndarray:
    """Preprocess an AVFAD sustained-vowel signal.

    The sequence is mono conversion, DC removal, optional crop to the first
    repetition, edge trimming, and RMS normalization.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform.
    fs : int
        Sampling rate in Hz.
    crop_to_one_rep : bool, default=True
        Whether to keep only the first detected vowel repetition.

    Returns
    -------
    numpy.ndarray
        Preprocessed mono waveform as ``float32``.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, 0]
    x = x - np.mean(x)
    if crop_to_one_rep:
        rep_end, _, _ = find_first_repetition(x, fs)
        x = x[:rep_end]
    x = trim_edges(x, fs)
    x = rms_normalize(x)
    return x.astype(np.float32)


def read_nsp_signal(nsp_path: Path) -> tuple[np.ndarray, int]:
    """Read an NSP file and convert its signal to float32 audio."""
    fs, x = nspfile.read(str(nsp_path))  # this function returns only a tuple (fs, x) due to the kwargs # type: ignore
    fs = int(fs)
    x = np.asarray(x)

    if len(x) == 0:
        raise RuntimeError("Empty signal")

    # NSP files may already contain float audio otherwise scale by the observed peak so integer-like samples fit the
    # standard [-1, 1] audio range. Note that we do not try to detect the original bit depth (e.g., 16-bit) as it is not
    # guaranteed to be consistent across files.
    x = x.astype(np.float32)
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x /= peak

    return x, fs


def f0_computable(
    x: np.ndarray,
    fs: int,
    sex: Literal["m", "f"] | None,
) -> bool:
    """Check whether the mean F0 is computable for a given signal.

    Parameters
    ----------
    x : numpy.ndarray
        Input waveform.
    fs : int
        Sampling rate in Hz.
    sex : Literal["m", "f"] | None
        Sex of the speaker (normalized).
    Returns
    -------
    bool
        Whether the mean F0 is computable.
    """

    fmin, fmax = f0_range_for_sex(sex)

    sound = parselmouth.Sound(x, sampling_frequency=int(fs))
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)

    pitch_data = pitch.selected_array["frequency"]
    pitch_data[pitch_data == 0] = np.nan

    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    return not np.isnan(mean_f0) and mean_f0 > 0.0


# ============================================================
# ======================= SVD DATASET ========================
# ============================================================

SVD_PATTERN = re.compile(r"(?P<id>\d+)-(?P<vowel>[a-z]+)_(?P<intonation>[a-z]+)\.nsp")

def process_svd_metadata(path: Path) -> Metadata:
    parsed = SVD_PATTERN.fullmatch(path.name)
    if not parsed:
        raise ValueError(f"Filename does not match SVD pattern: {path.name}")
    data = parsed.groupdict()
    session_id = int(data["id"])
    try:
        metadata = svd_metadata_by_session[session_id].copy()
    except KeyError as exc:
        raise ValueError(f"Metadata not found for session ID: {data["id"]}") from exc

    # change all keys to be strings for typing consistency
    metadata = {str(k): v for k, v in metadata.items()}

    # rename dict keys to match the target CSV schema
    metadata["age"] = int(metadata.pop("talkerage"))
    metadata["talkerid"] = f"SVD_{str(metadata["talkerid"]).strip()}"

    metadata["vowel"] = data["vowel"]
    metadata["intonation"] = data["intonation"]
    pathologies = metadata["pathologies"]
    metadata["pathologies"] = "healthy" if pd.isna(pathologies) else str(pathologies).strip() or "healthy"
    metadata["talkersex"] = normalize_sex(metadata.pop("talkersex"))
    metadata.pop("diagnosis")
    metadata.pop("comment")

    metadata["file"] = path.name
    metadata["dataset"] = "SVD"
    metadata["vowel"] = data["vowel"]
    metadata["intonation"] = data["intonation"]

    metadata = Metadata(**metadata)  # validate and convert to target schema

    return metadata


def process_file(
    dataset: str,
    sound_path: Path,
    idx: int,
) -> tuple[int, list[Metadata] | list[Never]]:

    # get metadata
    metadata: Metadata
    if dataset == "SVD":
        metadata = process_svd_metadata(sound_path)
    elif dataset == "AVFAD":
        raise NotImplementedError("AVFAD processing not implemented yet")
    elif dataset == "VOICED":
        raise NotImplementedError("VOICED processing not implemented yet")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # filter out records with drop flag
    if sound_path.name in to_drop[to_drop["database"] == dataset]["filename"].values:
        return idx, []

    # read and preprocess signal - this is separated as some dataset could in provide multple records per file
    if dataset == "SVD":
        x, fs = read_nsp_signal(sound_path)
        x = preprocess_signal(x, fs)

        metadata.id = idx
        metadata.computable_f0 = f0_computable(x, fs, metadata.talkersex)
        metadata_list = [metadata]
        wav_name = f"record_{idx:04d}.wav"

        sf.write(OUT_DIR / wav_name, x, fs)
        idx += 1
    elif dataset == "AVFAD":
        raise NotImplementedError("AVFAD processing not implemented yet")
    elif dataset == "VOICED":
        raise NotImplementedError("VOICED processing not implemented yet")

    return idx, metadata_list


# ============================================================
# Public API
# ============================================================


def build_dataset():

    dataset_metadata = []
    idx = 0

    # SVD
    for nsp_path in tqdm(list(SVD_ROOT.rglob("*.nsp")), desc="Processing SVD"):
        idx, metadata = process_file(dataset="SVD", sound_path=nsp_path, idx=idx)
        dataset_metadata.extend(metadata)

    # VOICED
    # for p in tqdm(list(VOICED_ROOT.rglob("*.dat")), desc="Processing VOICED"):
    #     idx = process_voiced_file(p, writer, idx)

    # avfad_meta = load_avfad_metadata(AVFAD_XLSX)

    # for r in tqdm(avfad_meta, desc="Processing AVFAD"):
    #     for suffix, vowel in AVFAD_FILES.items():
    #         p = find_avfad_wav(r["file_id"], suffix)
    #         if p is None:
    #             continue
    #         idx = process_avfad_file(p, r, vowel, writer, idx)

    dataset_metadata = pd.DataFrame(dataset_metadata)
    dataset_metadata.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    build_dataset()
