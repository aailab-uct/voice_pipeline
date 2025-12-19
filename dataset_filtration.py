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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X, y and groups for sklearn pipeline.

    Labeling:
        healthy       -> 0
        pathological  -> 1

    Parameters
    ----------
    excluded_pathologies : set[str]
        Patologie, které budou vyhozeny.
    include_age : bool
        Zda přidat věk do X.
    include_sex : bool
        Zda přidat pohlaví do X.
    allowed_vowels : set[str] or None
        Povolené samohlásky (např. {"a", "i"}). None = bez filtru.
    allowed_intonations : set[str] or None
        Povolené intonace (např. {"n"}). None = bez filtru.
    selected_datasets : set[str] or None
        Povolené datasety (např. {"SVD", "VOICED"}). None
    selected_fs : int or None
        Pokud není None, přeresampluje audio na tuto vzorkovací frekvenci.

    Returns
    -------
    X : np.ndarray (dtype=object)
        Každý prvek: [signal, (age), (sex)]
    y : np.ndarray
        Binární label (0 = healthy, 1 = pathological)
    groups : np.ndarray
        talkerID (pro GroupKFold).
    """
    if excluded_pathologies is None:
        excluded_pathologies = set()
    X = []
    y = []
    groups = []

    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in tqdm(rows, desc="Building X, y, groups"):
        pathology = row["pathologies"]

        # --- filter unwanted pathologies
        if pathology in excluded_pathologies:
            continue

        # --- filter vowel
        if allowed_vowels is not None and row["vowel"] not in allowed_vowels:
            continue

        # --- filter intonation
        if allowed_intonations is not None and row["intonation"] not in allowed_intonations:
            continue

        # --- filter dataset
        if selected_datasets is not None and row["dataset"] not in selected_datasets:
            continue

        # --- filter nan f0
        if drop_nan_f0 and row.get("nan_f0") == "True":
            continue

        # --- load audio
        original_file = AUDIO_DIR / f"record_{row["id"]}.wav"
        cached_file = AUDIO_DIR / f"record_{row["id"]}_{selected_fs}.wav" if selected_fs is not None else original_file
        if cached_file.exists():
            signal, fs = sf.read(cached_file, dtype="float32")
        elif original_file.exists():
            signal, fs = sf.read(original_file, dtype="float32")
            if selected_fs is not None and fs != selected_fs:
                signal = librosa.resample(signal, orig_sr=fs, target_sr=selected_fs, res_type="soxr_vhq")
                fs = selected_fs
                signal = signal.astype(np.float32)
                sf.write(AUDIO_DIR / f"record_{row["id"]}_{selected_fs}.wav", data=signal, samplerate=selected_fs)
        else:
            print(f"Warning: audio file for record {row["id"]} not found, skipping.")
            continue

        # --- build feature vector

        features = [signal]

        if include_age:
            age = int(row["age"]) if row["age"] else np.nan
            features.append(age)

        if include_sex:
            match row["talkersex"].lower():
                case "f" | "w" | "female" | "woman":
                    sex = 1
                case "m" | " male" | "man":
                    sex = 0
                case _:
                    sex = np.nan
                    print(f"Warning: unknown sex '{row["talkersex"]}' for record {row["id"]}")
            features.append(sex)

        # --- binary label by health status
        is_healthy = row["pathologies"].lower() == "healthy"
        label = 0 if is_healthy else 1

        X.append(features)
        y.append(label)
        groups.append((row["talkerid"]))

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    groups = np.array(groups, dtype=str)

    return X, y, groups


# ============================================================
# CLI entry point
# ============================================================


def main():
    X, y, groups = build_dataset(
        excluded_pathologies={"Sängerstimme", "Gesangsstimme"},
        include_age=True,
        include_sex=True,
        allowed_vowels={"a"},
        allowed_intonations={"n"},
    )

    print(X.shape, y.shape, groups.shape)


if __name__ == "__main__":
    main()
