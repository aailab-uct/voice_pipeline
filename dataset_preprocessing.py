"""Module for preprocessing raw datasets into a standardized format."""

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

# declaration of output directories
OUT_DIR = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)

OUT_CSV = OUT_DIR / "metadata.csv"


# ============================================================
# Target CSV schema (FIXED)
# ============================================================

# template for new metadata file
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
# ======================= SVD DATASET ========================
# ============================================================

# session id´s from SVD dataste which are known for corruption, artifacts or other problems
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
}  # added new, 713 missing HEDR/HDR8 subchunk
# home directory for SVD database recordings
SVD_ROOT = Path("data", "SVD")
# path to file with metadata
SVD_METADATA_CSV = SVD_ROOT / "svd_information.csv"
# pattern by which recordings are filtered, pattern: ID-VOWEL_INTONATION.nsp
SVD_PATTERN = re.compile(r"(?P<id>\d+)-(?P<vowel>[a-z]+)_(?P<intonation>[a-z]+)\.nsp")


def load_svd_metadata(csv_path: Path):
    """
    Function which take input csv file and make
    dictionary like list of keys for easy access

    Args:
        csv_path: path to csv file
    Returns:
        list: list of dictionary like keys for accessing of  input data
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def normalize_pathology(value):
    """
    Function for normalization of labeling of pathologies
    from none to healthy or from empty label to healthy.

    Args:
        value: label
    Returns:
        str: normalized label
    """
    # None version
    if value is None:
        return "healthy"
    # missing version
    value = value.strip()
    # return label if repaired or original label was already healthy
    return value if value else "healthy"


def process_svd_file(nsp_path: Path, metadata, writer, record_idx):
    """
    Main function for processing a SVD file, which is run in cykle through directory with recordings.

    Args:
        nsp_path: path to nsp file
        metadata: list with dictionary like structure with metadata
        writer: output csv file
        record_idx: index of recording
    Returns:

    """
    # check if pattern is correct and extract  id, vowel and intonation
    m = SVD_PATTERN.fullmatch(nsp_path.name)
    if not m:
        return record_idx

    parts = m.groupdict()
    speaker_id = parts["id"]

    # connect id from name of recording with speaker id in metadata and
    # return dict of relevant metadata or none if dosnt find match
    src = next((m for m in metadata if m.get("talkerid") == speaker_id), None)
    if not src:
        return record_idx

    # check if id is not None and is not in list of excluded files
    session_id = src.get("sessionid")
    if session_id is not None and int(session_id) in SESSION_ID_TO_DELETE:
        return record_idx

    # new name for recording generation
    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"

    # try to load nsp file wia nspfile library
    try:
        # sampling rate and data
        fs, x = nspfile.read(nsp_path)  # type: ignore
    # wide exception for nsp file load
    except RuntimeError as e:
        print(f"Warning: failed to read {nsp_path}: {e}")
        return record_idx
    # conversion od sample rate to int
    fs = int(fs)
    # conversion of data to nummpy array
    x = np.array(x, dtype=np.float32)

    # cut first moments of the start and end of each recording
    x, _ = librosa.effects.trim(x, top_db=15, frame_length=256 * fs // 50000, hop_length=65 * fs // 50000)
    # save preprocessed file to wav file with new unificated name
    sf.write(OUT_DIR / wav_name, x, fs)

    # Check if f0 can be extracted
    fmin = 60.0
    fmax = 250.0

    # creation of praat object sound from first channel of data
    sound = parselmouth.Sound(values=x[:, 0], sampling_frequency=fs)
    # call calculation of f0 component with defined limits of frequency
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)

    # final call of calculation of mean from frequencies
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    # structure of new row which is writen in new metadata file
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
        "nan_f0": np.isnan(mean_f0),
    }

    # writing of metadata row
    writer.writerow(row)
    return record_idx + 1


# ============================================================
# ====================== VOICED DATASET ======================
# ============================================================

# home repository for voiced dataset
VOICED_ROOT = Path("data") / "VOICED"
# sampling rate of voiced dataset recordings
TARGET_FS = 8000


def extract_number(text):
    """
    Function for extraction of numbers from text
    Args:
        str: input text
    Returns:
        str: extracted number

    """
    m = re.search(r"\d+", text)
    return m.group() if m else None


def parse_voiced_info(info_path: Path):
    """
    Function which goes through rows of input file with metadata
    and transforms them to dictionary structure
    Args:
        info_path: path to file with metadata (for every recording)
    Returns:
        dict: dictionary with metadata

    """
    # empty dict
    meta = {}
    # open file with metadata
    with open(info_path, encoding="utf-8") as f:
        # goes through lines of input file
        for line in f:
            # separate files by :
            if ":" in line:
                k, v = line.split(":", 1)
                # reconstruct dictionary
                meta[k.strip().lower()] = v.strip()
    return meta


def process_voiced_file(dat_path: Path, writer, record_idx):
    """
    Main function for prosessing voiced recordings.
    Include loading recordings, converting to numpy array, triming,
    checking if is possible to calculate f0 and saving recording and metadata
    Args:
        dat_path: path to directory with recordings for voiced dataset
        writer: output csv file
        record_idx: index of recording
    Returns:
        None
    """

    # stem command takes last component of path without suffix
    base = dat_path.stem
    # create new path which is the same as before except name of file which is updated
    info_path = dat_path.with_name(f"{base}-info.txt")

    # if it failed somewhere, return id
    if not info_path.exists():
        return record_idx

    # open wfdb record, before that separate suffix from file and converts it to string
    record = wfdb.rdrecord(str(dat_path.with_suffix("")))
    # change shape from (x, 1) to (x,)
    x = record.p_signal.squeeze()  # type: ignore
    # return sampling rate of record and converts it to int
    fs = int(record.fs)  # type: ignore
    # conversion to numpy array
    x = np.array(x, dtype=np.float32)

    # trim of moments on the begining and end of signal
    x, _ = librosa.effects.trim(x, top_db=15, frame_length=256 * fs // 50000, hop_length=65 * fs // 50000)

    # Check if f0 can be extracted
    # limits of frequency for extraction
    fmin = 60.0
    fmax = 250.0

    # creation of parselmouth obeject sound
    sound = parselmouth.Sound(x, sampling_frequency=fs)
    # call of function to calculate f0 via parselmouth
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)

    # separation of frequency from calculated f0 values
    pitch_data = pitch.selected_array["frequency"]
    # replacement of zero values to nan (where it was not able to calculate f0)
    pitch_data[pitch_data == 0] = np.nan

    # call of function to calculate mean fo f0 frquencies
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    # genration of new name of record
    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"
    # saving preprocessed record as wav file
    sf.write(OUT_DIR / wav_name, x, fs)

    # call of function that transforms file with metadata to dictionary
    info = parse_voiced_info(info_path)

    # definition of structure of metadata file
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
        "nan_f0": np.isnan(mean_f0),
    }

    # write command for new metadata of preprocessed recording
    writer.writerow(row)
    return record_idx + 1


# ============================================================
# ======================= AVFAD DATASET =======================
# ============================================================

# data layout:
# data/AVFAD/
#   AVFAD_01_00_00.xlsx
#   A/
#   B/
#   C/
#   ...
# inside letter folders are wavs like ZZZ001.wav, ZZZ002.wav, ZZZ003.wav

AVFAD_ROOT = Path("data") / "AVFAD"
AVFAD_XLSX = AVFAD_ROOT / "AVFAD_01_00_00.xlsx"
AVFAD_SHEET = "AVFAD"

# first three recordings per row
AVFAD_FILES = {
    "001": "i",
    "002": "a",
    "003": "u",
}

# optional crop to one repetition
AVFAD_CROP_TO_ONE_REP = True


def load_avfad_metadata(xlsx_path: Path):
    """
    Reads AVFAD excel and returns list of dicts with needed fields.
    Uses columns:
    File ID, Age, Sex, CMVD-I Dimension 1 (word system)
    """

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[AVFAD_SHEET]

    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    col = {str(h).strip(): i for i, h in enumerate(header) if h is not None}

    need = ["File ID", "Age", "Sex", "CMVD-I Dimension 1 (word system)"]
    for k in need:
        if k not in col:
            raise ValueError(f"AVFAD: missing column '{k}' in sheet '{AVFAD_SHEET}'")

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
    """
    Finds file like {File ID}{suffix}.wav anywhere under AVFAD_ROOT (including letter subfolders)
    Example: ZZZ + 001 -> ZZZ001.wav
    """
    name = f"{file_id}{suffix}.wav"
    return next(AVFAD_ROOT.rglob(name), None)


def crop_to_first_big_interval_from_zero(x: np.ndarray, fs: int, intervals: np.ndarray, min_len_s: float = 1.0) -> np.ndarray:
    """
    Crop from start (0) to the end of the first "big" voiced interval.
    intervals: array [[start, end], ...] in samples.
    """
    if x.size == 0 or intervals is None or len(intervals) == 0:
        return x

    min_len = int(min_len_s * fs)

    for s, e in intervals:
        if (e - s) >= min_len:
            end = min(len(x), int(e))
            return x[:end]

    # fallback: if no big interval found, keep original
    return x


def process_avfad_file(wav_path: Path, src_row: dict, vowel: str, writer, record_idx: int) -> int:
    """
    Processes single AVFAD wav into unified wav name and writes one metadata row.
    """
    x, fs = sf.read(wav_path, dtype=np.float32)

    # if stereo, keep first channel
    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x[:, 0]

    # optional crop
    if AVFAD_CROP_TO_ONE_REP:
        intervals = librosa.effects.split(x, top_db=25, frame_length=2048, hop_length=512)
        x = crop_to_first_big_interval_from_zero(x, int(fs), intervals, min_len_s=1.0)

    # trim, same style as other functions
    x, _ = librosa.effects.trim(x, top_db=15, frame_length=256 * fs // 50000, hop_length=65 * fs // 50000)

    # f0 check
    fmin = 60.0
    fmax = 250.0
    sound = parselmouth.Sound(x, sampling_frequency=int(fs))
    pitch = call(sound, "To Pitch", 0.0, fmin, fmax)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    record_id = f"{record_idx:04d}"
    wav_name = f"record_{record_id}.wav"
    sf.write(OUT_DIR / wav_name, x, int(fs))

    p = normalize_pathology(str(src_row.get("pathology")) if src_row.get("pathology") is not None else None)
    if p.strip().lower() == "normal":
        p = "healthy"

    row = {
        "id": record_id,
        "sessionid": None,
        "talkerid": f"AVFAD_{src_row.get('file_id')}",
        "talkersex": src_row.get("sex"),
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
# Public API function
# ============================================================


def build_dataset():
    """
    Function that call other functions which preprocessed dataset
    Args:
        None
    Returns:
        None
    """
    # load of SVD metadata
    svd_metadata = load_svd_metadata(SVD_METADATA_CSV)

    # all it starts with creation of new svd file for metadata
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        # definition of output place and metadata csv structure
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        # starting index
        idx = 0

        # run for SVD dataset
        for p in tqdm(list(SVD_ROOT.rglob("*.nsp")), desc="Processing SVD"):
            idx = process_svd_file(p, svd_metadata, writer, idx)

        # run for VOICED dataset
        for p in tqdm(list(VOICED_ROOT.rglob("*.dat")), desc="Processing VOICED"):
            idx = process_voiced_file(p, writer, idx)

        # run for AVFAD dataset
        avfad_meta = load_avfad_metadata(AVFAD_XLSX)
        for r in tqdm(avfad_meta, desc="Processing AVFAD"):
            for suffix, vowel in AVFAD_FILES.items():
                p = find_avfad_wav(r["file_id"], suffix)
                if p is None:
                    continue
                idx = process_avfad_file(p, r, vowel, writer, idx)


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    build_dataset()
