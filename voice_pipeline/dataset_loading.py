"""Load filtered waveforms and labels for experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
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
    allowed_vowels: set[str] | str | None = "a",
    allowed_intonations: set[str] | str | None = None,
    selected_datasets: set[str] | str | None = None,
    selected_fs: int | None = None,
    drop_nan_f0: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build arrays used by the sklearn pipeline.

    Healthy samples are encoded as 0 and pathological samples as 1.

    Parameters
    ----------
    excluded_pathologies : set of str or None, default=None
        Pathology labels to remove from the dataset.
    include_age : bool, default=True
        Whether to include age as a feature column.
    include_sex : bool, default=True
        Whether to include sex as a feature column.
    allowed_vowels : set of str or str or None, default="a"
        Allowed vowels, for example ``{"a", "i"}``. ``None`` disables this filter.
    allowed_intonations : set of str or str or None, default="n"
        Allowed intonations, for example ``{"n"}``. ``None`` disables this filter.
    selected_datasets : set of str or str or None, default=None
        Allowed dataset names, for example ``{"SVD", "VOICED"}``. ``None`` disables this filter.
    selected_fs : int or None, default=None
        Target sampling rate in Hz. ``None`` keeps the original sampling rate.
    drop_nan_f0 : bool, default=True
        Whether to drop rows marked as having invalid F0.

    Returns
    -------
    X : numpy.ndarray of dtype object
        Feature rows. The first column is always the waveform, followed by
        optional age and sex columns.
    y : numpy.ndarray
        Binary label (0 = healthy, 1 = pathological)
    groups : numpy.ndarray
        Talker IDs used for group-aware cross-validation.
    pathologies : numpy.ndarray
        Pathology name per sample.
    datasets : numpy.ndarray
        Dataset name per sample, for example ``"SVD"`` or ``"AVFAD"``.
    """

    if excluded_pathologies is None:
        excluded_pathologies = set()
    if type(allowed_vowels) is str:
        allowed_vowels = {allowed_vowels}
    if allowed_vowels is not None:
        if type(allowed_vowels) is str:
            allowed_vowels = {allowed_vowels}

    X = []
    y = []
    groups = []
    pathologies_out = []
    datasets_out = []

    rows = pd.read_csv(METADATA_CSV).to_dict("records")

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

        record_id = row["id"]
        original_file = AUDIO_DIR / f"record_{record_id:04d}.wav"
        cached_file = AUDIO_DIR / f"record_{record_id:04d}_{selected_fs}.wav" if selected_fs is not None else original_file
        if cached_file.exists():
            signal, fs = sf.read(cached_file, dtype="float32")
        elif original_file.exists():
            signal, fs = sf.read(original_file, dtype="float32")
            if selected_fs is not None and fs != selected_fs:
                signal = librosa.resample(signal, orig_sr=fs, target_sr=selected_fs, res_type="soxr_vhq")
                fs = selected_fs
                signal = signal.astype(np.float32)
                sf.write(AUDIO_DIR / f"record_{record_id}_{selected_fs}.wav", data=signal, samplerate=selected_fs)
        else:
            print(f"Warning: audio file for record {record_id} not found, skipping.")
            continue

        features = [signal]

        if include_sex:
            match row["talkersex"].lower():  # TODO: this probably should already be normalized at his point
                case "f" | "w" | "female" | "woman":
                    sex = 1
                case "m" | " male" | "man":
                    sex = 0
                case _:
                    sex = np.nan
                    print(f"Warning: unknown sex '{row['talkersex']}' for record {record_id}")
            features.append(sex)

        if include_age:
            age = int(row["age"]) if row["age"] else np.nan
            features.append(age)


        is_healthy = row["pathologies"].lower() == "healthy"
        label = 0 if is_healthy else 1

        X.append(features)
        y.append(label)
        groups.append(row["talkerid"])
        pathologies_out.append(pathology)
        datasets_out.append(row["dataset"])

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    groups = np.array(groups, dtype=str)
    pathologies_out = np.array(pathologies_out, dtype=str)
    datasets_out = np.array(datasets_out, dtype=str)

    return X, y, groups, pathologies_out, datasets_out


# ============================================================
# Testing
# ============================================================


def main():
    X, y, groups, _, _ = build_dataset(
        excluded_pathologies={"Sängerstimme", "Gesangsstimme"},
        include_age=True,
        include_sex=True,
        allowed_vowels={"a"},
        allowed_intonations={"n"},
    )
    print(X.shape, y.shape, groups.shape)


if __name__ == "__main__":
    main()
