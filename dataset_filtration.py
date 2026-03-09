"""
Module for filtering dataset and building feature and label arrays.
"""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm
import librosa


# ============================================================
# Paths
# ============================================================

AUDIO_DIR = Path("dataset")
METADATA_CSV = Path("dataset/metadata.csv")


# ============================================================
# Public API
# ============================================================

def build_dataset(
    excluded_pathologies: set[str] | None = None,
    include_age: bool = True,
    include_sex: bool = True,
    allowed_vowels: set[str] | None = None,
    allowed_intonations: set[str] | None = None,
    selected_datasets: set[str] | None = None,
    selected_fs: int | None = None,
    drop_nan_f0: bool = True,
    return_dataset_names: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X, y and groups for sklearn pipeline.

    Labeling:
        healthy       -> 0
        pathological  -> 1

    Returns
    -------
    X : np.ndarray (dtype=object)
        Each element: [signal, (age), (sex)]
    y : np.ndarray
        Binary label (0 = healthy, 1 = pathological)
    groups : np.ndarray
        talkerID (for GroupKFold)
    pathologies_out : np.ndarray
        string pathology per sample
    dataset_names : np.ndarray (optional)
        string dataset name per sample (only if return_dataset_names=True)
    """
    if excluded_pathologies is None:
        excluded_pathologies = set()

    X = []
    y = []
    groups = []
    pathologies_out = []
    dataset_names = []

    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="Building X, y, groups"):
        pathology = row["pathologies"]

        if pathology in excluded_pathologies:
            continue

        if allowed_vowels is not None and row["vowel"] not in allowed_vowels:
            continue

        if allowed_intonations is not None and row["intonation"] not in allowed_intonations:
            continue

        if selected_datasets is not None and row["dataset"] not in selected_datasets:
            continue

        if drop_nan_f0 and row.get("nan_f0") == "True":
            continue

        rec_id = row["id"]

        original_file = AUDIO_DIR / f"record_{rec_id}.wav"
        cached_file = (
            AUDIO_DIR / f"record_{rec_id}_{selected_fs}.wav"
            if selected_fs is not None
            else original_file
        )

        if cached_file.exists():
            signal, fs = sf.read(cached_file, dtype="float32")
        elif original_file.exists():
            signal, fs = sf.read(original_file, dtype="float32")
            if selected_fs is not None and fs != selected_fs:
                signal = librosa.resample(
                    signal, orig_sr=fs, target_sr=selected_fs, res_type="soxr_vhq"
                ).astype(np.float32)
                fs = selected_fs
                sf.write(
                    AUDIO_DIR / f"record_{rec_id}_{selected_fs}.wav",
                    data=signal,
                    samplerate=selected_fs,
                )
        else:
            print(f"Warning: audio file for record {rec_id} not found, skipping.")
            continue

        # --- PŘIDAT SEM ---
        duration_ms = len(signal) / fs * 1000
        if duration_ms < 100:
            print(f"Krátká nahrávka: record_{rec_id}, {duration_ms:.0f}ms, dataset={row['dataset']}")
            continue
        # --- KONEC ---

        features = [signal]

        if include_age:
            age = int(row["age"]) if row["age"] else np.nan
            features.append(age)

        if include_sex:
            s = (row.get("talkersex") or "").strip().lower()
            if s in {"f", "w", "female", "woman"}:
                sex = 1
            elif s in {"m", "male", "man"}:
                sex = 0
            else:
                sex = np.nan
                print(f"Warning: unknown sex '{row.get('talkersex')}' for record {rec_id}")
            features.append(sex)

        is_healthy = row["pathologies"].lower() == "healthy"
        label = 0 if is_healthy else 1

        X.append(features)
        y.append(label)
        groups.append(row["talkerid"])
        pathologies_out.append(pathology)

        if return_dataset_names:
            dataset_names.append(row["dataset"])

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    groups = np.array(groups, dtype=str)
    pathologies_out = np.array(pathologies_out, dtype=str)

    if return_dataset_names:
        dataset_names = np.array(dataset_names, dtype=str)
        return X, y, groups, pathologies_out, dataset_names

    return X, y, groups, pathologies_out


# ============================================================
# CLI entry point
# ============================================================

def main():
    X, y, groups, pathologies, dataset_names = build_dataset(
        excluded_pathologies={"Sängerstimme", "Gesangsstimme"},
        include_age=True,
        include_sex=True,
        allowed_vowels={"a"},
        allowed_intonations={"n"},
        return_dataset_names=True,
    )
    print(X.shape, y.shape, groups.shape, pathologies.shape, dataset_names.shape)


if __name__ == "__main__":
    main()