# Konfigurace pipeline

Konfigurace je rozdělena do tematických JSON souborů. Každý soubor odpovídá jedné fázi pipeline.

---

## experiment.json

Metadata experimentu, nastavení datasetu a cross-validace.

### experiment
| Parametr | Typ | Popis |
|---|---|---|
| `experiment_name` | string | Název experimentu (použit v názvech výstupních souborů) |
| `experiment_description` | string | Volný popis experimentu |

### dataset
| Parametr | Typ | Popis |
|---|---|---|
| `selected_datasets` | list | Datasety k použití — `SVD`, `AVFAD`, `VOICED`, `PVQD` |
| `selected_fs` | int | Vzorkovací frekvence v Hz |
| `allowed_vowels` | list | Povolené samohlásky — např. `["a", "i", "u"]` |
| `allowed_intonations` | list | Povolené intonace — např. `["n"]` (neutrální) |
| `include_age` | bool | Zahrnout věk pacienta jako příznak |
| `include_sex` | bool | Zahrnout pohlaví pacienta jako příznak |
| `excluded_pathologies` | list | Patologie vyřazené z datasetu |

### cross_validation
| Parametr | Typ | Popis |
|---|---|---|
| `cv.type` | string | Typ cross-validace — `StratifiedGroupKFold` |
| `cv.n_splits` | int | Počet foldů |
| `scoring_mode` | string | Režim scoringu — `multimetric` nebo `single` |
| `refit_metric` | string | Metrika pro výběr nejlepšího modelu — např. `uar` |
| `n_repeats` | int | Počet opakování CV (`0` = vypnuto) |

---

## raw_preprocessing.json

Zpracování surového audio signálu před extrakcí příznaků. Obsahuje dvě nezávislé fáze.

### raw_augmentation
Modifikuje každý existující waveform (train i test). Nemění počet vzorků.

| Parametr | Typ | Popis |
|---|---|---|
| `enabled` | bool | Zapnutí/vypnutí celého bloku |
| `sr` | int | Vzorkovací frekvence |
| `noise` | bool | Aditivní Gaussův bílý šum |
| `gain` | bool | Náhodné škálování amplitudy |
| `polarity` | bool | Inverze znaménka signálu (50% pravděpodobnost) |
| `time_crop` | bool | Náhodný ořez + zero-pad na původní délku |
| `pitch_shift` | bool | Posun výšky tónu (zachová délku) — pomalé |
| `time_stretch` | bool | Změna rychlosti (zachová výšku) — pomalé |
| `reverb` | bool | Konvoluce se syntetickou impulsní odezvou místnosti |
| `codec` | bool | Simulace telefonního kodeku (mu-law) |
| `noise_std` | float | Směrodatná odchylka šumu |
| `gain_min` | float | Minimální faktor zesílení |
| `gain_max` | float | Maximální faktor zesílení |
| `crop_min_frac` | float | Minimální podíl délky signálu po ořezu |
| `pitch_semitones_range` | [float, float] | Rozsah posunu výšky v půltónech |
| `stretch_rate_range` | [float, float] | Rozsah faktoru změny rychlosti |
| `reverb_decay` | float | Útlum reverbu |
| `random_state` | int | Seed pro reprodukovatelnost |

### waveform_synthesis
Generuje zcela nové waveformy ze stávajících. Běží pouze na trénovacím foldu — syntetické audio nikdy nevstupuje do testovacího foldu ani do metrik.

| Parametr | Typ | Popis |
|---|---|---|
| `enabled` | bool | Zapnutí/vypnutí celého bloku |
| `sr` | int | Vzorkovací frekvence |
| `noise` | bool | Aditivní šum |
| `gain` | bool | Náhodné škálování amplitudy |
| `polarity` | bool | Inverze znaménka |
| `time_crop` | bool | Náhodný ořez |
| `pitch_shift` | bool | Posun výšky tónu — pomalé |
| `time_stretch` | bool | Změna rychlosti — pomalé |
| `reverb` | bool | Reverb |
| `codec` | bool | Simulace kodeku |
| `noise_std` | float | Směrodatná odchylka šumu |
| `gain_min` | float | Minimální faktor zesílení |
| `gain_max` | float | Maximální faktor zesílení |
| `random_state` | int | Seed pro reprodukovatelnost |

---

## feature_extraction.json

Přepínače a parametry jednotlivých extraktorů příznaků. Každý extraktor lze nezávisle zapnout/vypnout.

### Společné parametry (platí pro každý extraktor)
| Parametr | Typ | Popis |
|---|---|---|
| `enabled` | bool | Zapnutí/vypnutí extraktoru |
| `sr` | int | Vzorkovací frekvence |

### mfcc
| Parametr | Typ | Popis |
|---|---|---|
| `n_mfcc` | int | Počet mel-frekvenčních kepstrálních koeficientů |

### timedomain
| Parametr | Typ | Popis |
|---|---|---|
| `use_zcr` | bool | Zahrnout zero crossing rate |
| `use_entropy` | bool | Zahrnout entropii signálu |
| `use_skewness` | bool | Zahrnout šikmost (skewness) |
| `use_diff_pitch` | bool | Zahrnout diferenci základní frekvence |

### zcr
Samostatný extraktor zero crossing rate (bez dalších parametrů).

### spectral
Spektrální příznaky (centroid, bandwidth, rolloff, flux, ...). Bez dalších parametrů.

### opensmile
| Parametr | Typ | Popis |
|---|---|---|
| `feature_set` | string | Sada příznaků — `GeMAPS` nebo `ComParE_2016` |
| `feature_level` | string | Úroveň extrakce — `functionals` nebo `lld` |

### f0deriv
Derivace základní frekvence F0. Bez dalších parametrů.

### nonlinear
Nelineární dynamické příznaky. Bez dalších parametrů.

### vocaltract
| Parametr | Typ | Popis |
|---|---|---|
| `lp_order` | int | Řád lineární predikce (LPC) pro modelování vokálního traktu |

---

## feature_preprocessing.json

Zpracování příznaků po extrakci, před resamplingem.

### feature_augmentation
Přidává Gaussův šum do příznaků (augmentace ve feature space, ne v audio).

| Parametr | Typ | Popis |
|---|---|---|
| `enabled` | bool | Zapnutí/vypnutí |
| `std_factor` | float | Faktor směrodatné odchylky šumu (relativně k std příznaku) |
| `random_state` | int | Seed pro reprodukovatelnost |

### feature_selection
| Parametr | Typ | Popis |
|---|---|---|
| `enabled` | bool | Zapnutí/vypnutí |
| `selector.type` | string | Typ selektoru — `SelectKBest`, `SelectFromModel`, `RFECV` |
| `selector.score_func` | string | Skórovací funkce pro SelectKBest — `mutual_info_classif`, `f_classif` |
| `selector.k` | int | Počet příznaků pro SelectKBest |

---

## resampling.json

Vyvážení tříd na trénovacím foldu. Syntetické vzorky nikdy nevstupují do testovacího foldu.

| Parametr | Typ | Popis |
|---|---|---|
| `resampler.type` | string | Typ resampleru — `SMOTESampler`, `VAESampler` |
| `resampler.enabled` | bool | Zapnutí/vypnutí |
| `resampler.random_state` | int | Seed pro reprodukovatelnost |
| `resampler.artificial_only` | bool | Použít pouze syntetické vzorky (bez originálních) |
| `resampler.sampling_strategy` | string | Strategie vyvážení — `auto`, `minority`, `not majority` |

---

## classifier.json

Definice klasifikátoru a mřížky hyperparametrů pro ruční grid search (alternativa k `search.json`).

### classifier
| Parametr | Typ | Popis |
|---|---|---|
| `type` | string | Třída klasifikátoru — `RandomForestClassifier`, `KANClassifier`, ... |
| `random_state` | int | Seed pro reprodukovatelnost |
| `n_jobs` | int | Počet paralelních vláken (`-1` = všechna dostupná) |

### param_grid
Seznam slovníků s explicitními hodnotami pro grid search (scikit-learn formát). Používá se pouze pokud `search.json` má `backend: gridsearch` nebo `halving`. Pro Optuna/RandomSearch jsou relevantní rozsahy v `search.json`.

---

## search.json

Konfigurace hyperparametrové optimalizace. Sekce `search` definuje backend a jeho parametry, sekce `param_space` definuje co se optimalizuje.

### search
| Parametr | Typ | Popis |
|---|---|---|
| `backend` | string | Metoda optimalizace — `optuna`, `gridsearch`, `randomsearch`, `halving` |
| `metric` | string | Optimalizovaná metrika — `uar`, `accuracy`, `f1`, ... |
| `direction` | string | Směr optimalizace — `maximize` nebo `minimize` |
| `n_trials` | int | Počet triálů (Optuna, RandomSearch) |
| `sampler` | string | Sampler pro Optuna — `TPE`, `CmaEs`, `Random`, `Grid` |
| `pruner` | string | Pruner pro Optuna — `median`, `hyperband`, `none` |
| `n_iter` | int | Počet iterací pro RandomSearch |
| `factor` | int | Redukční faktor pro Halving |
| `n_jobs` | int | Počet paralelních vláken |
| `timeout` | int/null | Časový limit v sekundách (`null` = bez limitu) |
| `random_state` | int | Seed pro reprodukovatelnost |

### param_space
Každý klíč odpovídá parametru pipeline ve tvaru `krok__parametr`. Hodnota definuje prostor prohledávání.

| Typ | Klíče | Popis |
|---|---|---|
| `categorical` | `choices` | Diskrétní výčet hodnot |
| `int` | `low`, `high` | Celé číslo v rozsahu |
| `float` | `low`, `high` | Desetinné číslo v rozsahu |
| `log_float` | `low`, `high` | Desetinné číslo v log měřítku (vhodné pro learning rate) |

---

## visualization.json

Přepínače pro generování grafů po dokončení experimentu.

| Parametr | Typ | Popis |
|---|---|---|
| `plot_uar_per_pathology` | bool | Graf UAR pro každou patologii zvlášť |
| `plot_feature_histograms` | bool | Histogramy distribucí příznaků |
