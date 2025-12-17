import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

from imblearn.pipeline import Pipeline as ImbPipeline

from feature_extraction import MFCCExtractor, DropSignalColumn
from signal_augmenters import RawAugmenter
from feature_augmenters import FeatureAugmenter
from oversampling import RandomCopySampler, SMOTESampler
from dataset_filtration import build_dataset

# ==========
# 1) Build imblearn pipeline
# ==========


def build_pipeline():
    """
    imblearn Pipeline:

    RawAugment (waveform) ->
    MFCCExtractor ->
    StandardScaler ->
    FeatureAugment (optional jitter) ->
    SMOTE (train-only resampling) ->
    LogisticRegression
    """
    pipe = ImbPipeline(
        steps=[
            ("raw_aug", RawAugmenter(enabled=True, noise_std=0.005, random_state=42)),
            ("mfcc", MFCCExtractor(sr=16000, n_mfcc=20)),
            ("drop_sig", DropSignalColumn()),
            ("scaler", StandardScaler()),
            ("feat_aug", FeatureAugmenter(enabled=True, std_factor=0.02, random_state=42)),
            ("resampler", SMOTESampler(enabled=True, random_state=42, artificial_only=True)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    return pipe


# ==========
# 2) Minimal synthetic audio dataset with groups
# ==========


def make_synthetic_audio_dataset(n_records=60, sr=16000, random_state=0):
    """
    Create a toy dataset:
    - X: each sample is one "recording" and demographic features
    - y: binary classification (healthy/pathological)
    - groups: speaker IDs (1 group per sample)
    """
    rng = np.random.RandomState(random_state)

    X = []
    y = []
    groups = []

    for i in range(n_records):
        label = rng.randint(0, 2)  # binary classes 0 / 1 - healthy/pathological
        sex = rng.choice([0, 1])  # sex feature: 0/1 - male/female
        length = rng.randint(int(0.5 * sr), int(1.5 * sr))  # 0.5–1.5 s
        base = 0.0 if label == 0 else 0.3  # class-dependent mean shift
        age = rng.randint(20, 60)  # random age between 20 and 60
        sig = base + 0.1 * rng.randn(length).astype(np.float32)
        X.append([sig, sex, age])
        y.append(label)
        groups.append(i)  # each recording has its own group ID

    X = np.array(X, dtype=object)
    y = np.array(y)
    groups = np.array(groups)

    return X, y, groups


# ==========
# 3) Run GroupKFold CV (leakage-safe at recording level)
# ==========

if __name__ == "__main__":
    # 1) Generate synthetic dataset
    X, y, groups = build_dataset(
        excluded_pathologies={"Sängerstimme", "Gesangsstimme"},
        include_age=True,
        include_sex=True,
        allowed_vowels={"a"},
        allowed_intonations={"n"},
        selected_datasets={"SVD", "VOICED"},
        selected_fs=8000,
    )
    print(X.shape, y.shape, groups.shape)

    # 1.1) Encode groups to number
    le = LabelEncoder()
    groups = le.fit_transform(groups)

    # 2) Build pipeline
    pipe = build_pipeline()

    # Example: toggle augmentation behaviour
    # pipe.set_params(raw_aug__enabled=False)
    # pipe.set_params(feat_aug__enabled=False)

    # 3) GroupKFold CV to avoid leakage between recordings
    cv = GroupKFold(n_splits=5)

    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        groups=groups,  # <- key point: split by recording ID
        scoring="accuracy",
        n_jobs=-1,
    )

    print("CV accuracy:", scores.mean(), "+/-", scores.std())
