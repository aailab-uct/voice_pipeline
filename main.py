import numpy as np
import librosa

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ==========
# 1) Custom transformers
# ==========

class DropSignalColumn(BaseEstimator, TransformerMixin):
    """
    Drops the first column (waveform) from the input data.
    Useful for testing feature-level pipeline without raw audio.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        assert X[0,0].ndim == 1, "First column must be waveform"
        return self

    def transform(self, X):
        return np.asarray(X[:,1:], dtype=np.float32)

class RawAugmenter(BaseEstimator, TransformerMixin):
    """
    Waveform-level augmentation:
    - random gain
    - additive Gaussian noise

    Shape-preserving: does NOT change number of samples.
    """
    def __init__(self, enabled=True, noise_std=0.005,
                 gain_min=0.8, gain_max=1.2, random_state=None):
        self.enabled = enabled
        self.noise_std = noise_std
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.random_state = random_state

    def fit(self, X, y=None):
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def transform(self, X):
        if not self.enabled:
            # Keep as object array to allow variable-length signals
            return X

        X_out = []
        for row in X:
            x = row[0]  # the first column is the waveform
            x = np.asarray(x, dtype=np.float32)

            # random gain
            gain = self.rng_.uniform(self.gain_min, self.gain_max)
            x_aug = x * gain

            # additive noise
            noise = self.rng_.normal(0.0, self.noise_std, size=x.shape)
            x_aug = x_aug + noise

            row[0] = x_aug.astype(np.float32)

            X_out.append(row)

        return np.array(X_out, dtype=object)


class MFCCExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts MFCC features: mean and std over time for each coefficient.
    Output: 2 * n_mfcc features per sample.
    """
    def __init__(self, sr=16000, n_mfcc=20, pass_through=False):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.pass_through = pass_through

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for row in X:
            x = row[0]  # the first column is the waveform
            x = np.asarray(x, dtype=np.float32)
            mfcc = librosa.feature.mfcc(y=x, sr=self.sr, n_mfcc=self.n_mfcc)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            if self.pass_through:
                feat = np.concatenate([row, mfcc_mean, mfcc_std])
            else:
                feat = np.concatenate([row[1:], mfcc_mean, mfcc_std])
            feats.append(feat)
        if self.pass_through:
            return np.array(feats, dtype=object)
        return np.vstack(feats)


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Feature-level augmentation: small Gaussian jitter relative to per-feature std.

    Note: This is a standard transformer, so it is applied to both
    train and test folds. It does NOT change n_samples, only perturbs
    existing features (no leakage because it does not use y).
    """
    def __init__(self,
                 enabled=False,
                 std_factor=0.01,
                 apply_to_all=False,
                 random_state=None):
        self.enabled = enabled
        self.std_factor = std_factor
        self.random_state = random_state
        self.apply_to_all = apply_to_all

    def fit(self, X, y=None):
        if self.apply_to_all:
            X = np.asarray(X, dtype=np.float32)
            self.mask = np.ones(X.shape[1], dtype=bool)
        else:
            self.mask = np.sum(np.abs(np.remainder(X, 1)), axis=0) == 0
            X = np.asarray(X[:,self.mask], dtype=np.float32)
        self.X_std_ = np.std(X, axis=0, ddof=1)
        self.rng_ = np.random.RandomState(self.random_state)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if not self.enabled:
            return X
        noise_std = self.std_factor * (self.X_std_ + 1e-8)
        noise = self.rng_.normal(loc=0.0, scale=noise_std, size=X.shape)
        X[:,self.mask] += noise
        return X


# ==========
# 2) Build imblearn pipeline
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
    pipe = ImbPipeline(steps=[
        ("raw_aug", RawAugmenter(enabled=True, noise_std=0.005, random_state=42)),
        ("mfcc", MFCCExtractor(sr=16000, n_mfcc=20, pass_through=True)),
        ("drop_sig", DropSignalColumn()),
        ("scaler", StandardScaler()),
        ("feat_aug", FeatureAugmenter(enabled=True, std_factor=0.02, random_state=42)),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    return pipe


# ==========
# 3) Minimal synthetic audio dataset with groups
# ==========

def make_synthetic_audio_dataset(n_records=60, sr=16000, random_state=0):
    """
    Create a toy dataset:
    - each sample is one "recording": random noise with class-dependent mean
    - groups = recording IDs (1 group per sample)
    """
    rng = np.random.RandomState(random_state)

    X = []
    y = []
    groups = []

    for i in range(n_records):
        label = rng.randint(0, 2)  # binary classes 0 / 1
        length = rng.randint(int(0.5 * sr), int(1.5 * sr))  # 0.5–1.5 s
        base = 0.0 if label == 0 else 0.3  # class-dependent mean shift
        age = rng.randint(20, 60)  # random age between 20 and 60
        sig = base + 0.1 * rng.randn(length).astype(np.float32)
        X.append([sig,age])
        y.append(label)
        groups.append(i)  # each recording has its own group ID

    X = np.array(X, dtype=object)
    y = np.array(y)
    groups = np.array(groups)

    return X, y, groups


# ==========
# 4) Run GroupKFold CV (leakage-safe at recording level)
# ==========

if __name__ == "__main__":
    # 1) Generate synthetic dataset
    X, y, groups = make_synthetic_audio_dataset()
    print(X.shape, y.shape, groups.shape)

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
        groups=groups,      # <- key point: split by recording ID
        scoring="accuracy",
        n_jobs=-1
    )

    print("CV accuracy:", scores.mean(), "+/-", scores.std())
