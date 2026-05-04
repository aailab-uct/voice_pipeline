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

### Dostupné backendy

#### optuna
Bayesovská optimalizace — učí se z předchozích triálů a cílí na slibné oblasti prostoru. Nejefektivnější pro velké nebo spojité prostory.

| Parametr | Typ | Popis |
|---|---|---|
| `metric` | string | Optimalizovaná metrika — `uar`, `accuracy`, `f1`, ... |
| `direction` | string | `maximize` nebo `minimize` |
| `n_trials` | int | Celkový počet triálů |
| `sampler` | string | Algoritmus vzorkování — viz tabulka níže |
| `pruner` | string | Algoritmus pruning — viz tabulka níže |
| `n_jobs` | int | Počet paralelních triálů |
| `timeout` | int/null | Časový limit v sekundách (`null` = bez limitu) |
| `random_state` | int | Seed pro reprodukovatelnost |

**Samplery:**
| Hodnota | Popis |
|---|---|
| `TPE` | Tree-structured Parzen Estimator — výchozí, dobře funguje pro většinu úloh |
| `CmaEs` | CMA-ES — vhodné pro spojité prostory s mnoha parametry |
| `Random` | Čistě náhodné vzorkování (bez učení) |
| `Grid` | Vyčerpávající mřížka (jako gridsearch, ale přes Optuna) |
| `QMC` | Quasi-Monte Carlo — rovnoměrnější pokrytí než Random |
| `NSGAII` | Genetický algoritmus pro multi-objective optimalizaci |

**Prunery** (předčasné zastavení slabých triálů):
| Hodnota | Popis |
|---|---|
| `median` | Zastaví trial pokud je pod mediánem ostatních ve stejném kroku |
| `hyperband` | Hyperband — agresivnější, vhodné pro velké počty triálů |
| `successive_halving` | Podobné Hyperbandu, jednodušší varianta |
| `none` | Bez pruningu — každý trial doběhne do konce |

---

#### gridsearch
Vyčerpávající prohledání všech kombinací z `choices`. Garantuje nalezení optima v dané mřížce, ale škáluje exponenciálně.

| Parametr | Typ | Popis |
|---|---|---|
| `metric` | string | Optimalizovaná metrika |
| `refit` | bool | Přetrénovat nejlepší model na celém trénovacím setu |
| `n_jobs` | int | Počet paralelních vláken |
| `verbose` | int | Úroveň výpisu (`0` = ticho, `2` = každá kombinace) |

> Používá pouze klíč `choices` z `param_space`. Klíče `low`/`high` jsou ignorovány.

---

#### randomsearch
Náhodné vzorkování kombinací — rychlejší než grid search, prakticky srovnatelné výsledky při dostatečném `n_iter`.

| Parametr | Typ | Popis |
|---|---|---|
| `metric` | string | Optimalizovaná metrika |
| `n_iter` | int | Počet náhodně vzorkovaných kombinací |
| `refit` | bool | Přetrénovat nejlepší model na celém trénovacím setu |
| `n_jobs` | int | Počet paralelních vláken |
| `verbose` | int | Úroveň výpisu |
| `random_state` | int | Seed pro reprodukovatelnost |

> Pokud je zadán `choices`, vzorkuje z výčtu. Pokud je zadán `low`/`high`, vzorkuje uniformně z rozsahu.

---

#### halving
Successive Halving — začíná s mnoha kandidáty na malém počtu dat, v každém kole eliminuje nejslabší polovinu a zdvojnásobí data. Výrazně rychlejší než gridsearch při srovnatelné kvalitě.

| Parametr | Typ | Popis |
|---|---|---|
| `metric` | string | Optimalizovaná metrika |
| `refit` | bool | Přetrénovat nejlepší model na celém trénovacím setu |
| `factor` | int | Redukční faktor — `3` znamená v každém kole přežije 1/3 kandidátů |
| `min_resources` | string/int | Minimální zdroje (vzorky) v prvním kole — `exhaust`, `smallest`, nebo číslo |
| `aggressive_elimination` | bool | Agresivnější eliminace při nedostatku kol |
| `n_jobs` | int | Počet paralelních vláken |
| `random_state` | int | Seed pro reprodukovatelnost |

> Používá `choices` stejně jako gridsearch. Kombinuje se s `HalvingGridSearchCV` nebo `HalvingRandomSearchCV`.

---

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

---

## dataset_builder.json

Konfigurace skriptu pro sestavení datasetu ze surových nahrávek. Týká se fáze před ML pipeline — konverze, ořez, normalizace a uložení do jednotného formátu.

### output
| Parametr | Typ | Popis |
|---|---|---|
| `out_dir` | string | Výstupní složka pro WAV soubory |
| `metadata_csv` | string | Cesta k výstupnímu CSV s metadaty |

### filtering
| Parametr | Typ | Popis |
|---|---|---|
| `min_age` | int | Minimální věk mluvčího — záznamy mladších mluvčích jsou přeskočeny |

### f0
| Parametr | Typ | Popis |
|---|---|---|
| `ranges.m` | [float, float] | Rozsah F0 pro muže v Hz — `[fmin, fmax]` |
| `ranges.f` | [float, float] | Rozsah F0 pro ženy v Hz — `[fmin, fmax]` |
| `default_range` | [float, float] | Záložní rozsah F0 pokud pohlaví není známé |

### signal_normalization
Společné parametry normalizace signálu platné pro všechny datasety, pokud není dataset-specifická hodnota definována.

| Parametr | Typ | Popis |
|---|---|---|
| `target_rms` | float | Cílová RMS hodnota po normalizaci |
| `min_rms` | float | Minimální RMS — signál s nižší hodnotou není normalizován (ochrana před dělením nulou) |
| `trim_top_db` | int | Práh v dB pro ořez ticha na začátku a konci signálu |

### datasets

Každý dataset má vlastní podsekci. Dataset-specifická hodnota `trim_top_db` přepisuje globální hodnotu z `signal_normalization`.

#### SVD
| Parametr | Typ | Popis |
|---|---|---|
| `root` | string | Kořenová složka datasetu |
| `metadata_csv` | string | Cesta k CSV s metadaty mluvčích |
| `session_ids_to_delete` | list[int] | Blacklist session ID — záznamy s tímto ID jsou přeskočeny |

#### PVQD
| Parametr | Typ | Popis |
|---|---|---|
| `root` | string | Kořenová složka datasetu |
| `demographics_xlsx` | string | Cesta k XLSX souboru s demografickými údaji |
| `trim_top_db` | int | Přepisuje globální `trim_top_db` pro tento dataset |

#### VOICED
| Parametr | Typ | Popis |
|---|---|---|
| `root` | string | Kořenová složka datasetu |
| `trim_top_db` | int | Přepisuje globální `trim_top_db` pro tento dataset |
| `frame_length_factor` | int | Čitatel pro výpočet `frame_length = factor * fs // reference_fs` |
| `hop_length_factor` | int | Čitatel pro výpočet `hop_length = factor * fs // reference_fs` |
| `reference_fs` | int | Referenční vzorkovací frekvence ze které byly faktory odvozeny |

#### AVFAD
| Parametr | Typ | Popis |
|---|---|---|
| `root` | string | Kořenová složka datasetu |
| `xlsx` | string | Cesta k XLSX souboru s metadaty |
| `sheet` | string | Název listu v XLSX souboru |
| `crop_to_one_rep` | bool | Oříznout nahrávku na první repetici samohlásky |
| `noise_est_s` | float | Délka okna pro odhad šumové hladiny v sekundách |
| `silence_ratio` | float | Podíl dynamického rozsahu určující práh ticha |
| `min_pause_s` | float | Minimální délka pauzy považované za hranici repetice (s) |
| `min_rep_s` | float | Minimální délka první repetice — kratší jsou ignorovány (s) |
| `window_s` | float | Délka framu pro výpočet energie signálu (s) |
| `files` | object | Mapování přípon souborů na samohlásky — `{"001": "i", "002": "a", ...}` |
