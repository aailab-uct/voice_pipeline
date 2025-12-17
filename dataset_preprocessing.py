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
]


# ============================================================
# Session IDs to exclude (SVD only)
# ============================================================

SESSION_ID_TO_DELETE = {
    1573, 87, 1372, 2088, 2077, 2070,
    2081, 2426, 2428, 2448, 2450, 2451, 2457
}


# ============================================================
# ======================= SVD DATASET ========================
# ============================================================

SVD_ROOT = Path("data")
SVD_METADATA_CSV = SVD_ROOT / "SVD" / "svd_information.csv"

SVD_PATTERN = re.compile(
    r"(?P<id>\d+)-(?P<vowel>[a-z]+)_(?P<intonation>[a-z]+)\.nsp"
)


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

    src = next(
        (m for m in metadata if m.get("talkerid") == speaker_id),
        None
    )
    if not src:
        return record_idx

    session_id = src.get("sessionid")
    if session_id is not None and int(session_id) in SESSION_ID_TO_DELETE:
        return record_idx

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"

    fs, x = nspfile.read(nsp_path)
    x, _ = librosa.effects.trim(
        x,
        top_db=15,
        frame_length=256,
        hop_length=65
    )
    sf.write(OUT_DIR / wav_name, x, fs)

    row = {
        "id": record_id,
        "sessionid": src.get("sessionid"),
        "talkerid": f"SVD_{src.get('talkerid')}",
        "talkersex": src.get("talkersex"),
        "age": src.get("talkerage"),
        "pathologies": normalize_pathology(src.get("pathologies")),
        "file": nsp_path.name,
        "vowel": parts["vowel"],
        "intonation": parts["intonation"],
        "dataset": "SVD",
    }

    writer.writerow(row)
    return record_idx + 1


# ============================================================
# ====================== VOICED DATASET ======================
# ============================================================

VOICED_ROOT = Path("data") / "VOICED"
TARGET_FS = 16000


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

    record = wfdb.rdrecord(str(dat_path.with_suffix("")))
    x = record.p_signal.squeeze()
    fs = record.fs  # VOICED = 8000 Hz

    # --- resample to 16 kHz
    if fs != TARGET_FS:
        x = librosa.resample(x, orig_sr=fs, target_sr=TARGET_FS)
        fs = TARGET_FS

    x, _ = librosa.effects.trim(
        x,
        top_db=15,
        frame_length=256,
        hop_length=65
    )

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"
    sf.write(OUT_DIR / wav_name, x, fs)

    info = parse_voiced_info(info_path)

    row = {
        "id": record_id,
        "sessionid": None,
        "talkerid": f"VOICED_{extract_number(base)}",
        "talkersex": info.get("gender"),
        "age": info.get("age"),
        "pathologies": info.get("diagnosis"),
        "file": dat_path.name,
        "vowel": "a",
        "intonation": "n",
        "dataset": "VOICED",
    }

    writer.writerow(row)
    return record_idx + 1


# ============================================================
# Public API function
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


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    build_dataset()
