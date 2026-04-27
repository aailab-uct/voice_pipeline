# Voice Pipeline

# TODO -> rewrite as this is AI generated

Root files are the researcher-facing entry points:

- `main.py` - experiment protocol: load config, load data, run CV, save outputs.
- `dataset_preprocessing.py` - build the standardized `dataset/metadata.csv` and WAV files.
- `config.py`, `config_experimental.py`, `config.yaml` - experiment settings.

Reusable implementation code lives in `voice_pipeline/`:

- `dataset_loading.py` - filters metadata and loads waveforms for experiments.
- `pipeline.py` - builds the sklearn/imblearn pipeline.
- `feature_extraction.py` - acoustic feature extractors.
- `signal_augmentation.py`, `feature_augmentation.py` - augmentation steps.
- `sampling.py` - oversampling methods.
- `classifiers.py` - custom sklearn-compatible classifiers.
- `reporting.py`, `visualization.py` - experiment outputs.
- `utils.py`, `feature_cache.py`, `hashable.py` - shared helpers.
