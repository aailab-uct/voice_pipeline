"""
Module for preprocessing raw datasets into a standardized format (BASELINE).

Baseline = no bandpass, no preemphasis, no global RMS normalization, no peak safety.
Only:
- load audio
- convert to mono float
- trim silence
- (AVFAD only) optionally crop first repetition
- save WAV + metadata
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
# Preprocessing parameters
# ============================================================

# True  = ořezat ticho na začátku i na konci nahrávky
# False = ořezat ticho pouze na konci nahrávky (začátek zachován)
TRIM_LEADING_SILENCE = False

# True  = normalizovat amplitudu na rozsah [-1, 1] (peak normalizace)
# False = ponechat původní amplitudu
NORMALIZE_AMPLITUDE = True


def normalize_amplitude(x: np.ndarray) -> np.ndarray:
    """Peak normalizace signálu do rozsahu [-1, 1]."""
    if not NORMALIZE_AMPLITUDE:
        return x
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
        mmax = float(np.max(np.abs(x))) if x.size else 0.0
        if mmax > 0.0:
            x /= mmax
    return x


def trim_silence(x: np.ndarray, top_db: float = 15, **kwargs) -> np.ndarray:
    """Ořízne ticho na konci (vždy) a volitelně i na začátku nahrávky."""
    if TRIM_LEADING_SILENCE:
        x, _ = librosa.effects.trim(x, top_db=top_db, **kwargs)
    else:
        # Ořez pouze konce: obrátíme signál, ořežeme začátek (= původní konec), obrátíme zpět
        x_rev = x[::-1]
        x_rev, _ = librosa.effects.trim(x_rev, top_db=top_db, **kwargs)
        x = x_rev[::-1]
    return np.asarray(x, dtype=np.float32)


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
    if age is None:
        return False
    try:
        return float(age) >= MIN_AGE
    except Exception:
        return False


# ============================================================
# F0 configuration
# ============================================================

F0_RANGES = {
    "m": (75.0, 300.0),
    "f": (100.0, 600.0),
}

DEFAULT_F0_RANGE = (60.0, 250.0)


def normalize_sex_mf(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"m", "male", "man"}:
        return "m"
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
        pid = r[col.get("Participant ID", col.get("Participant ID "))]
        if pid is None:
            continue
        out.append(
            {
                "participant_id": str(pid).strip(),
                "sex": r[col.get("Gender")],
                "age": r[col.get("Age")],
                "diagnosis": r[col.get("Diagnosis", col.get("Diagnosis "))],
            }
        )
    return out


def normalize_pvqd_pathology(value):
    if value is None:
        return "healthy"
    s = str(value).strip()
    if not s:
        return "healthy"
    s_low = s.lower()
    if s_low in {"n", "normal"}:
        return "healthy"
    if s_low == "y":
        return "pathology"
    s = re.sub(r"\s*,\s*", ", ", s.strip())
    s = re.sub(r"\s+", " ", s)
    return s


def pvqd_is_a_vowel_file(fname: str, participant_id: str) -> bool:
    f = fname.lower()
    pid = participant_id.lower()
    if pid not in f:
        return False
    base = Path(f).stem
    if re.search(rf"(^|[_-])a([_-]|$)", base):
        return True
    if "vowel" in base and re.search(r"vowel[_-]?a", base):
        return True
    return False


def _pvqd_load_prepare_signal(wav_path: Path, src_row: dict) -> tuple[np.ndarray, int] | None:
    if not is_adult(src_row.get("age")):
        return None

    x, fs = sf.read(wav_path, dtype=np.float32)
    fs = int(fs)

    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x[:, 0]

    x = normalize_amplitude(x)
    x = trim_silence(x, top_db=15)
    return x, fs


def process_pvqd_file(wav_path: Path, src_row: dict, writer, record_idx: int) -> int:
    prepared = _pvqd_load_prepare_signal(wav_path, src_row)
    if prepared is None:
        return record_idx

    x, fs = prepared

    sex_mf = normalize_sex_mf(src_row.get("sex"))
    fmin, fmax = f0_range_for_sex(sex_mf)

    sound = parselmouth.Sound(x, sampling_frequency=fs)
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"
    sf.write(OUT_DIR / wav_name, x, fs)

    row = {
        "id": record_id,
        "sessionid": None,
        "talkerid": f"PVQD_{src_row.get('participant_id')}",
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
    1573, 87, 1372, 2088, 2077, 2070, 2081, 2426, 2428, 2448, 2450, 2451, 2457, 713,
}

SVD_ROOT = Path("data", "SVD")
SVD_METADATA_CSV = SVD_ROOT / "svd_information.csv"
SVD_PATTERN = re.compile(r"(?P<id>\d+)-(?P<vowel>[a-z]+)_(?P<intonation>[a-z]+)\.nsp")


def load_svd_metadata(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_pathology(value):
    if value is None:
        return "healthy"
    value = value.strip()
    return value if value else "healthy"


def _svd_load_prepare_signal(nsp_path: Path, metadata) -> tuple[np.ndarray, int, dict, dict] | None:
    m = SVD_PATTERN.fullmatch(nsp_path.name)
    if not m:
        return None
    parts = m.groupdict()
    speaker_id = parts["id"]

    src = next((mm for mm in metadata if mm.get("talkerid") == speaker_id), None)
    if not src:
        return None

    if not is_adult(src.get("talkerage")):
        return None

    session_id = src.get("sessionid")
    if session_id is not None and int(session_id) in SESSION_ID_TO_DELETE:
        return None

    try:
        fs, x = nspfile.read(nsp_path)
    except RuntimeError as e:
        print(f"Warning: failed to read {nsp_path}: {e}")
        return None

    fs = int(fs)
    x = np.asarray(x)

    if x.ndim > 1:
        x = x.mean(axis=1)

    x = normalize_amplitude(x)
    x = trim_silence(x, top_db=15, frame_length=256 * fs // 50000, hop_length=65 * fs // 50000)
    return x, fs, src, parts


def process_svd_file(nsp_path: Path, metadata, writer, record_idx):
    prepared = _svd_load_prepare_signal(nsp_path, metadata)
    if prepared is None:
        return record_idx

    x, fs, src, parts = prepared

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"
    sf.write(OUT_DIR / wav_name, x, fs)

    sex_mf = normalize_sex_mf(src.get("talkersex"))
    fmin, fmax = f0_range_for_sex(sex_mf)

    sound = parselmouth.Sound(x, sampling_frequency=fs)
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    row = {
        "id": record_id,
        "sessionid": src.get("sessionid"),
        "talkerid": f"SVD_{src.get('talkerid')}",
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


def _voiced_load_prepare_signal(dat_path: Path) -> tuple[np.ndarray, int, dict, str] | None:
    base = dat_path.stem
    info_path = dat_path.with_name(f"{base}-info.txt")
    if not info_path.exists():
        return None

    info = parse_voiced_info(info_path)
    if not is_adult(info.get("age")):
        return None

    record = wfdb.rdrecord(str(dat_path.with_suffix("")))
    x = record.p_signal.squeeze()
    fs = int(record.fs)
    x = np.asarray(x, dtype=np.float32)

    x = normalize_amplitude(x)
    x = trim_silence(x, top_db=15, frame_length=256 * fs // 50000, hop_length=65 * fs // 50000)
    return x, fs, info, base


def process_voiced_file(dat_path: Path, writer, record_idx):
    prepared = _voiced_load_prepare_signal(dat_path)
    if prepared is None:
        return record_idx

    x, fs, info, base = prepared

    sex_mf = normalize_sex_mf(info.get("gender"))
    fmin, fmax = f0_range_for_sex(sex_mf)

    sound = parselmouth.Sound(x, sampling_frequency=fs)
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
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


def _avfad_load_prepare_signal(wav_path: Path, src_row: dict) -> tuple[np.ndarray, int, str, str | None] | None:
    if not is_adult(src_row.get("age")):
        return None

    x, fs = sf.read(wav_path, dtype=np.float32)
    fs = int(fs)

    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x[:, 0]

    x = normalize_amplitude(x)
    x = trim_silence(x, top_db=15)

    if AVFAD_CROP_TO_ONE_REP:
        intervals = librosa.effects.split(x, top_db=20)
        if len(intervals):
            a, b = intervals[0]
            x = x[a:b]

    x = np.asarray(x, dtype=np.float32)

    sex_mf = normalize_sex_mf(src_row.get("sex"))
    p = normalize_pathology(str(src_row.get("pathology")) if src_row.get("pathology") is not None else None)
    if p.strip().lower() == "normal":
        p = "healthy"

    return x, fs, p, sex_mf


def process_avfad_file(wav_path: Path, src_row: dict, vowel: str, writer, record_idx: int) -> int:
    prepared = _avfad_load_prepare_signal(wav_path, src_row)
    if prepared is None:
        return record_idx

    x, fs, p, sex_mf = prepared

    fmin, fmax = f0_range_for_sex(sex_mf)

    try:
        sound = parselmouth.Sound(x, sampling_frequency=fs)
        pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    except Exception:
        print("AVFAD skipped:", src_row.get("file_id"), wav_path.name, "fs=", fs, "len=", len(x) / fs)
        return record_idx

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"
    sf.write(OUT_DIR / wav_name, x, fs)

    row = {
        "id": record_id,
        "sessionid": None,
        "talkerid": f"AVFAD_{src_row.get('file_id')}",
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
                wp = find_avfad_wav(r["file_id"], suffix)
                if wp is None:
                    continue
                idx = process_avfad_file(wp, r, vowel, writer, idx)

        pvqd_meta = load_pvqd_metadata(PVQD_DEMOGRAPHICS_XLSX)
        for r in tqdm(pvqd_meta, desc="Processing PVQD"):
            pid = r["participant_id"]
            for p in PVQD_ROOT.rglob("*.wav"):
                if pvqd_is_a_vowel_file(p.name, pid):
                    idx = process_pvqd_file(p, r, writer, idx)


if __name__ == "__main__":
    build_dataset()