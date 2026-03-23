"""Module for preprocessing raw datasets into a standardized format.
- normalization metrics on raw signal are finalized only on SVD and AVFAD datasets
"""

# ============================================================
# Imports
# ============================================================

import csv
import re
from pathlib import Path

import nspfile
import soundfile as sf
import librosa
import wfdb
from tqdm import tqdm
import numpy as np
import parselmouth
from parselmouth.praat import call
import openpyxl


# ============================================================
# Output
# ============================================================

OUT_DIR = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)

OUT_CSV = OUT_DIR / "metadata.csv"


# ============================================================
# Target CSV schema (FIXED)
# ============================================================

CSV_COLUMNS = [
    "id",
    "sessionid",
    "talkerid",
    "talkersex",
    "age",
    "pathologies",
    "file",
    "vowel",
    "intonation",
    "dataset",
    "nan_f0",
]


# ============================================================
# Filtering
# ============================================================

MIN_AGE = 18


def is_adult(age) -> bool:
    """
    Return True if age is numeric and >= MIN_AGE. Otherwise False.
    Handles None, strings, floats safely.
    """
    if age is None:
        # TODO: We should probably handle this in one safer place, as otherwise we might miss errors in other steps
        return False
    try:
        return float(age) >= MIN_AGE
    except Exception:# TODO: dtto
        return False


# ============================================================
# F0 configuration
# ============================================================

F0_RANGES = {
    "m": (75.0, 300.0),
    "f": (100.0, 600.0),
}

DEFAULT_F0_RANGE = (60.0, 250.0)  # TODO: Why? this makes litlle sense if m starts at 75 and f at 100.


def normalize_sex_mf(value):  # TODO: duplicate code of feature_extraction.py
    """
    Normalize sex/gender to 'm' or 'f'.
    Returns None if unknown.
    """
    if value is None:
        return None

    s = str(value).strip().lower()

    if s in {"m", "male", "man"}:
        return "m"

    # SVD uses 'w' for women
    if s in {"f", "female", "woman", "w"}:
        return "f"

    if s == "0":
        return "m"

    if s == "1":
        return "f"

    return None


def f0_range_for_sex(sex_mf):
    return F0_RANGES.get(sex_mf, DEFAULT_F0_RANGE)


# ============================================================
# Shared signal preprocessing
# ============================================================

TRIM_TOP_DB = 20
TARGET_RMS  = 0.1
MIN_RMS     = 1e-6

# AVFAD crop parametry
NOISE_EST_S   = 0.1
SILENCE_RATIO = 0.3
MIN_PAUSE_S   = 0.15
MIN_REP_S     = 0.5
WINDOW_S      = 0.05


def frame_energy_db(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    n_frames = (len(x) - frame_length) // hop_length + 1
    if n_frames <= 0:
        return np.array([-100.0], dtype=np.float32)
    return np.array([
        10 * np.log10(np.mean(x[i * hop_length: i * hop_length + frame_length] ** 2) + 1e-10)
        for i in range(n_frames)
    ], dtype=np.float32)


def subtract_noise_energy(edb: np.ndarray, fs: int, hop_length: int) -> np.ndarray:
    n_noise = min(max(1, int(NOISE_EST_S * fs / hop_length)), len(edb))
    noise_level_db = float(np.median(edb[:n_noise]))
    return np.maximum(edb - noise_level_db, -100.0)


def adaptive_silence_threshold(edb: np.ndarray) -> float:
    noise_floor = float(np.percentile(edb, 10))
    voiced_peak = float(np.percentile(edb, 90))
    return noise_floor + SILENCE_RATIO * (voiced_peak - noise_floor)


def find_first_repetition(x: np.ndarray, fs: int) -> tuple[int, list[tuple[int, int]], float]:
    frame_length = max(4, int(WINDOW_S * fs))
    hop_length   = max(1, frame_length // 2)
    edb          = frame_energy_db(x, frame_length, hop_length)
    edb_clean    = subtract_noise_energy(edb, fs, hop_length)
    thresh       = adaptive_silence_threshold(edb_clean)
    silent       = edb_clean < thresh
    n_frames     = len(edb_clean)

    first_voiced = next((i for i, s in enumerate(silent) if not s), None)
    if first_voiced is None:
        return len(x), [], thresh

    pauses, in_pause, p_start = [], False, 0
    for i in range(first_voiced, n_frames):
        if silent[i] and not in_pause:
            p_start, in_pause = i, True
        elif not silent[i] and in_pause:
            if (i - p_start) * hop_length / fs >= MIN_PAUSE_S:
                pauses.append((p_start * hop_length, i * hop_length))
            in_pause = False
    if in_pause and (n_frames - p_start) * hop_length / fs >= MIN_PAUSE_S:
        pauses.append((p_start * hop_length, len(x)))

    rep_end = len(x)
    for pause_start, _ in pauses:
        if pause_start / fs >= MIN_REP_S:
            rep_end = pause_start
            break
    return rep_end, pauses, thresh


def trim_edges(x: np.ndarray, fs: int) -> np.ndarray:
    fl = max(1, int(0.02 * fs))
    hl = max(1, fl // 2)
    edb         = frame_energy_db(x, fl, hl)
    noise_floor = float(np.percentile(edb, 10))
    peak        = float(np.percentile(edb, 95))
    start_thr   = noise_floor + 0.25 * (peak - noise_floor)
    end_thr     = noise_floor + 0.18 * (peak - noise_floor)
    min_run     = max(1, int(0.06 * fs / hl))

    def first_run(mask, n):
        cnt = 0
        for i, v in enumerate(mask):
            cnt = cnt + 1 if v else 0
            if cnt >= n: return i - n + 1
        return None

    def last_run(mask, n):
        cnt = 0
        for i in range(len(mask) - 1, -1, -1):
            cnt = cnt + 1 if mask[i] else 0
            if cnt >= n: return i + n - 1
        return None

    s = first_run(edb >= start_thr, min_run)
    e = last_run(edb >= end_thr,   min_run)
    if s is None or e is None:
        return x
    start, end = s * hl, min(len(x), e * hl + fl)
    return x[start:end] if end > start else x


def rms_normalize(x: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2)))
    return x * (TARGET_RMS / rms) if rms > MIN_RMS else x


def preprocess_signal(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Obecný preprocessing (SVD, VOICED, PVQD):
    mono -> DC removal -> trim okrajů -> RMS normalizace.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, 0]
    x = x - np.mean(x)
    x = trim_edges(x, fs)
    x = rms_normalize(x)
    return x.astype(np.float32)


def preprocess_avfad_signal(x: np.ndarray, fs: int, crop_to_one_rep: bool = True) -> np.ndarray:
    """
    AVFAD preprocessing:
    mono -> DC removal -> crop 1. repetice -> trim okrajů -> RMS normalizace.
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


# ============================================================
# ======================= PVQD DATASET =======================
# ============================================================

PVQD_ROOT = Path("data") / "PVQD"
PVQD_DEMOGRAPHICS_XLSX = PVQD_ROOT / "Demographics.xlsx"


def load_pvqd_metadata(xlsx_path: Path):
    df = openpyxl.load_workbook(xlsx_path, data_only=True).active
    rows = list(df.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    col = {h: i for i, h in enumerate(header)}

    out = []
    for r in rows[1:]:
        pid = r[
            col.get("Participant ID", col.get("Participant ID "))
        ]  # TODO: doopravdy je tohle nutne? nestaci ten strip nahore?
        if pid is None:
            continue
        out.append(
            {
                "participant_id": str(pid).strip(),
                "sex": r[col.get("Gender")],
                "age": r[col.get("Age")],
                "diagnosis": r[col.get("Diagnosis", col.get("Diagnosis "))],  # TODO: dtto
            }
        )
    return out


def normalize_pvqd_pathology(value):
    if value is None:
        return "healthy"
    pathology_string = str(value).lower().strip()
    if not pathology_string:
        return "healthy"


    # PVQD sheet uses N for normal, and sometimes Y as "has pathology"
    if pathology_string in {"n", "normal"}:
        return "healthy"
    if pathology_string == "y":
        return "pathology"

    # keep original text but normalize whitespace around commas
    pathology_string = re.sub(r"\s*,\s*", ", ", pathology_string.strip())
    pathology_string = re.sub(r"\s+", " ", pathology_string)
    return pathology_string


def pvqd_is_a_vowel_file(fname: str, participant_id: str) -> bool:
    """
    Heuristic: filename must contain participant_id and an 'a' token.
    Works for common patterns like:
      BL01_a.wav, BL01-a.wav, BL01_vowel_a.wav, BL01_A.wav
    """
    # TODO: ta db je doopravdy tak spatne oanotovana ze je treba tolik kroku?
    f = fname.lower()
    pid = participant_id.lower()

    if pid not in f:
        return False

    base = Path(f).stem

    # accept token a separated by _ or -
    if re.search(r"(^|[_-])a([_-]|$)", base):
        return True

    # accept explicit 'vowel a' like patterns
    if "vowel" in base and re.search(r"vowel[_-]?a", base):
        return True

    return False


def process_pvqd_file(wav_path: Path, src_row: dict, writer, record_idx: int) -> int:

    if not is_adult(src_row.get("age")):
        return record_idx

    x, fs = sf.read(wav_path, dtype=np.float32)

    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x[:, 0]

    x, _ = librosa.effects.trim(x, top_db=15)

    sex_mf = normalize_sex_mf(src_row.get("sex"))
    fmin, fmax = f0_range_for_sex(sex_mf)

    sound = parselmouth.Sound(x, sampling_frequency=int(fs))
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"
    sf.write(OUT_DIR / wav_name, x, int(fs))

    row = {
        "id": record_id,
        "sessionid": None,
        "talkerid": f"PVQD_{src_row.get("participant_id")}",
        "talkersex": sex_mf,
        "age": src_row.get("age"),
        "pathologies": normalize_pvqd_pathology(src_row.get("diagnosis")),
        "file": wav_path.name,
        "vowel": "a",
        "intonation": "n",
        "dataset": "PVQD",
        "nan_f0": np.isnan(mean_f0),
    }

    writer.writerow(row)
    return record_idx + 1


# ============================================================
# ======================= SVD DATASET ========================
# ============================================================

SESSION_ID_TO_DELETE = {
    1573,
    87,
    1372,
    2088,
    2077,
    2070,
    2081,
    2426,
    2428,
    2448,
    2450,
    2451,
    2457,
    713,
}

SVD_ROOT = Path("data", "SVD")
SVD_METADATA_CSV = SVD_ROOT / "svd_information.csv"
SVD_PATTERN = re.compile(r"(?P<id>\d+)-(?P<vowel>[a-z]+)_(?P<intonation>[a-z]+)\.nsp")


def load_svd_metadata(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def normalize_pathology(value):
    if value is None:
        return "healthy"
    value = value.strip()
    return value if value else "healthy"


def process_svd_file(nsp_path: Path, metadata, writer, record_idx):

    m = SVD_PATTERN.fullmatch(nsp_path.name)
    if not m:
        return record_idx

    parts = m.groupdict()
    speaker_id = parts["id"]

    src = next((m for m in metadata if m.get("talkerid") == speaker_id), None)
    if not src:
        return record_idx

    if not is_adult(src.get("talkerage")):
        return record_idx

    session_id = src.get("sessionid")
    if session_id is not None and int(session_id) in SESSION_ID_TO_DELETE:
        return record_idx

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"

    try:
        fs, x = nspfile.read(nsp_path)
    except RuntimeError as e:
        print(f"Warning: failed to read {nsp_path}: {e}")
        return record_idx

    fs = int(fs)
    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
        m = float(np.max(np.abs(x))) if x.size else 0.0
        if m > 1.0:
            x /= m

    x = preprocess_signal(x, fs)

    sf.write(OUT_DIR / wav_name, x, fs)

    sex_mf = normalize_sex_mf(src.get("talkersex"))
    fmin, fmax = f0_range_for_sex(sex_mf)

    sound = parselmouth.Sound(x, sampling_frequency=fs)
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    row = {
        "id": record_id,
        "sessionid": src.get("sessionid"),
        "talkerid": f"SVD_{src.get("talkerid")}",
        "talkersex": sex_mf,
        "age": src.get("talkerage"),
        "pathologies": normalize_pathology(src.get("pathologies")),
        "file": nsp_path.name,
        "vowel": parts["vowel"],
        "intonation": parts["intonation"],
        "dataset": "SVD",
        "nan_f0": np.isnan(mean_f0),
    }

    writer.writerow(row)
    return record_idx + 1


# ============================================================
# ====================== VOICED DATASET ======================
# ============================================================

VOICED_ROOT = Path("data") / "VOICED"


def extract_number(text):
    m = re.search(r"\d+", text)
    return m.group() if m else None


def parse_voiced_info(info_path: Path):

    meta = {}
    with open(info_path, encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip()
    return meta


def process_voiced_file(dat_path: Path, writer, record_idx):

    base = dat_path.stem
    info_path = dat_path.with_name(f"{base}-info.txt")

    if not info_path.exists():
        return record_idx

    info = parse_voiced_info(info_path)

    if not is_adult(info.get("age")):
        return record_idx

    record = wfdb.rdrecord(str(dat_path.with_suffix("")))

    x = record.p_signal.squeeze()
    fs = int(record.fs)
    x = np.array(x, dtype=np.float32)

    x, _ = librosa.effects.trim(x, top_db=15, frame_length=256 * fs // 50000, hop_length=65 * fs // 50000)

    sex_mf = normalize_sex_mf(info.get("gender"))
    fmin, fmax = f0_range_for_sex(sex_mf)

    sound = parselmouth.Sound(x, sampling_frequency=fs)
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)

    pitch_data = pitch.selected_array["frequency"]
    pitch_data[pitch_data == 0] = np.nan

    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"

    sf.write(OUT_DIR / wav_name, x, fs)

    row = {
        "id": record_id,
        "sessionid": None,
        "talkerid": f"VOICED_{extract_number(base)}",
        "talkersex": sex_mf,
        "age": info.get("age"),
        "pathologies": info.get("diagnosis"),
        "file": dat_path.name,
        "vowel": "a",
        "intonation": "n",
        "dataset": "VOICED",
        "nan_f0": np.isnan(mean_f0),
    }

    writer.writerow(row)
    return record_idx + 1


# ============================================================
# ======================= AVFAD DATASET =======================
# ============================================================

AVFAD_ROOT = Path("data") / "AVFAD"
AVFAD_XLSX = AVFAD_ROOT / "AVFAD_01_00_00.xlsx"
AVFAD_SHEET = "AVFAD"

AVFAD_FILES = {
    "001": "i",
    "002": "a",
    "003": "u",
}

AVFAD_CROP_TO_ONE_REP = True


def load_avfad_metadata(xlsx_path: Path):

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[AVFAD_SHEET]

    rows = ws.iter_rows(values_only=True)
    header = next(rows)

    col = {str(h).strip(): i for i, h in enumerate(header) if h is not None}

    out = []
    for r in rows:

        file_id = r[col["File ID"]]
        if file_id is None:
            continue

        out.append(
            {
                "file_id": str(file_id).strip(),
                "age": r[col["Age"]],
                "sex": r[col["Sex"]],
                "pathology": r[col["CMVD-I Dimension 1 (word system)"]],
            }
        )

    return out


def find_avfad_wav(file_id: str, suffix: str) -> Path | None:

    name = f"{file_id}{suffix}.wav"
    return next(AVFAD_ROOT.rglob(name), None)


def process_avfad_file(wav_path: Path, src_row: dict, vowel: str, writer, record_idx: int) -> int:

    if not is_adult(src_row.get("age")):
        return record_idx

    x, fs = sf.read(wav_path, dtype=np.float32)
    fs = int(fs)

    x = preprocess_avfad_signal(x, fs, crop_to_one_rep=AVFAD_CROP_TO_ONE_REP)

    sex_mf = normalize_sex_mf(src_row.get("sex"))
    fmin, fmax = f0_range_for_sex(sex_mf)

    # bezpečný výpočet F0 (skip problematických souborů)
    try:
        sound = parselmouth.Sound(x, sampling_frequency=int(fs))
        pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    except Exception as e: # TODO: unused e
        print("AVFAD skipped:", src_row.get("file_id"), wav_path.name, "fs=", fs, "len=", len(x) / fs)
        return record_idx

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"

    sf.write(OUT_DIR / wav_name, x, fs)

    p = normalize_pathology(str(src_row.get("pathology")) if src_row.get("pathology") is not None else None)

    if p.strip().lower() == "normal":
        p = "healthy"

    row = {
        "id": record_id,
        "sessionid": None,
        "talkerid": f"AVFAD_{src_row.get("file_id")}",
        "talkersex": sex_mf,
        "age": src_row.get("age"),
        "pathologies": p,
        "file": wav_path.name,
        "vowel": vowel,
        "intonation": "n",
        "dataset": "AVFAD",
        "nan_f0": np.isnan(mean_f0),
    }

    writer.writerow(row)
    return record_idx + 1


# ============================================================
# Public API
# ============================================================


def build_dataset():

    svd_metadata = load_svd_metadata(SVD_METADATA_CSV)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        idx = 0

        for p in tqdm(list(SVD_ROOT.rglob("*.nsp")), desc="Processing SVD"):
            idx = process_svd_file(p, svd_metadata, writer, idx)

        for p in tqdm(list(VOICED_ROOT.rglob("*.dat")), desc="Processing VOICED"):
            idx = process_voiced_file(p, writer, idx)

        avfad_meta = load_avfad_metadata(AVFAD_XLSX)

        for r in tqdm(avfad_meta, desc="Processing AVFAD"):
            for suffix, vowel in AVFAD_FILES.items():
                p = find_avfad_wav(r["file_id"], suffix)
                if p is None:
                    continue
                idx = process_avfad_file(p, r, vowel, writer, idx)

        # PVQD
        pvqd_meta = load_pvqd_metadata(PVQD_DEMOGRAPHICS_XLSX)
        for r in tqdm(pvqd_meta, desc="Processing PVQD"):
            pid = r["participant_id"]
            for p in PVQD_ROOT.rglob("*.wav"):
                if pvqd_is_a_vowel_file(p.name, pid):
                    idx = process_pvqd_file(p, r, writer, idx)


if __name__ == "__main__":
    build_dataset()