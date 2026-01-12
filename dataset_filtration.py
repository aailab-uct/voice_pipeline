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

# path to preprocessed dataset and metadata
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
    include_age : bool
        Condition if add age to X
    include_sex : bool
        Condition if add sex to X
    allowed_vowels : set[str] or None
        Allowed vowels (for example. {"a", "i"}). None = Without filter.
    allowed_intonations : set[str] or None
        Allowed intonation (for example. {"n"}). None = Without filter.
    selected_datasets : set[str] or None
        Allowed datasets (for example. {"SVD", "VOICED"}). None
    selected_fs : int or None
        It will resample sampling frequency to int or if None, leave original.

    Returns
    -------
    X : np.ndarray (dtype=object)
        All elements: [signal, (age), (sex)]
    y : np.ndarray
        Binari label (0 = healthy, 1 = pathological)
    groups : np.ndarray
        talkerID (for GroupKFold).
    """

    # this replace None for set() which is empty set
    if excluded_pathologies is None:
        excluded_pathologies = set()

    # init of lists
    X = []
    y = []
    groups = []

    # loading of new csv metadata file
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # tqdm part which in the end show progress bar
    for row in tqdm(rows, desc="Building X, y, groups"):
        # re labeling of variable
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

        # original name of record
        original_file = AUDIO_DIR /f"record_{row["id"]}.wav"
        # name of recrod for resampled records
        cached_file = AUDIO_DIR / f"record_{row["id"]}_{selected_fs}.wav" if selected_fs is not None else original_file
        if cached_file.exists():
            # load of cached records
            signal, fs = sf.read(cached_file, dtype="float32")
        elif original_file.exists():
            # if there is not cached record, original will be used
            signal, fs = sf.read(original_file, dtype="float32")
            # control if there is difference betwene original sampling rate and resampled sampling rate
            if selected_fs is not None and fs != selected_fs:
                # function that resample signal
                signal = librosa.resample(signal, orig_sr=fs, target_sr=selected_fs, res_type="soxr_vhq")
                # unification of sampling rates
                fs = selected_fs
                # change of data type
                signal = signal.astype(np.float32)
                # write of resampled recording
                sf.write(AUDIO_DIR / f"record_{row["id"]}_{selected_fs}.wav", data=signal, samplerate=selected_fs)
        else:
            print(f"Warning: audio file for record {row["id"]} not found, skipping.")
            continue

        # --- build feature vector

        features = [signal]

        # if parameter include age os set
        if include_age:
            # then row age is converted to int but only if there is not nan
            age = int(row["age"]) if row["age"] else np.nan
            features.append(age)

        # if parameter include sex is set
        if include_sex:
            # then there is matching (inside convert talkersex to lower signs)
            match row["talkersex"].lower():
                # then there is multiple cases what can
                # happen, it simply unifi different types of sex marking
                case "f" | "w" | "female" | "woman":
                    sex = 1
                case "m" | " male" | "man":
                    sex = 0
                case _:
                    # simple warning when there isnt sex
                    sex = np.nan
                    print(f"Warning: unknown sex '{row["talkersex"]}' for record {row["id"]}")
            features.append(sex)

        # set binari label by pathologies
        is_healthy = row["pathologies"].lower() == "healthy"
        label = 0 if is_healthy else 1

        # append of features and labels
        X.append(features)
        y.append(label)
        # append of talkerid
        groups.append((row["talkerid"]))

    # conversion to numpy array
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    groups = np.array(groups, dtype=str)

    return X, y, groups


# ============================================================
# CLI entry point
# ============================================================


def main():
    """
    main function which build dataset from recordings to vectors with data by various filtration
    """
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
