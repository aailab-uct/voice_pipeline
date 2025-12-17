import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

AUDIO_DIR = Path("dataset")
METADATA_CSV = Path("dataset/metadata.csv")


# ============================================================
# Public API
# ============================================================

def build_X_y_groups(
    excluded_pathologies,
    include_age=True,
    include_sex=True,
    allowed_vowels=None,
    allowed_intonations=None,
):
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

    Returns
    -------
    X : np.ndarray (dtype=object)
        Každý prvek: [signal, (age), (sex)]
    y : np.ndarray
        Binární label (0 = healthy, 1 = pathological)
    groups : np.ndarray
        talkerID (pro GroupKFold).
    """

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

        wav_path = AUDIO_DIR / f"record_{row['id']}.wav"
        if not wav_path.exists():
            continue

        signal, _ = sf.read(wav_path, dtype="float32")

        features = [signal]

        if include_age:
            age = int(row["age"]) if row["age"] else 0
            features.append(age)

        if include_sex:
            sex = 1 if row["talkersex"] in {"f", "w", "female"} else 0
            features.append(sex)

        # --- binary label by health status
        is_healthy = row["pathologies"].lower() == "healthy"
        label = 0 if is_healthy else 1

        X.append(features)
        y.append(label)
        groups.append((row["talkerid"]))

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=int)
    groups = np.array(groups, dtype=int)

    return X, y, groups


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    X, y, groups = build_X_y_groups(
        excluded_pathologies={"Sängerstimme", "Gesangsstimme"},
        include_age=True,
        include_sex=True,
        allowed_vowels={"a"},
        allowed_intonations={"n"},
    )

    print(X.shape, y.shape, groups.shape)
