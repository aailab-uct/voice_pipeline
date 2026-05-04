# Konfigurace pipeline (config.yaml)

Každá sekce odpovídá jednomu kroku pipeline. Přítomnost sekce v configu znamená že krok je aktivní, absence znamená že se přeskočí. Parametry neuvedené v configu se vezmou z defaultů v kódu.

---

## experiment

| Parametr | Typ | Popis |
|---|---|---|
| `name` | string | Název experimentu, použit v názvech výstupních souborů |
| `description` | string | Volný popis experimentu |

---

## dataset

| Parametr | Typ | Popis |
|---|---|---|
| `selected` | list | Datasety k použití — `SVD`, `AVFAD`, `VOICED`, `PVQD` |
| `fs` | int | Vzorkovací frekvence v Hz |
| `vowels` | list | Povolené samohlásky — `a`, `i`, `u` |
| `intonations` | list | Povolené intonace — `n` (neutrální) |
| `include_age` | bool | Zahrnout věk mluvčího jako příznak |
| `include_sex` | bool | Zahrnout pohlaví mluvčího jako příznak |
| `excluded_pathologies` | list | Patologie vyřazené z datasetu |

---

## search

| Parametr | Typ | Popis |
|---|---|---|
| `backend` | string | Metoda optimalizace — `optuna`, `gridsearch`, `randomsearch` |
| `n_trials` | int | Počet triálů (Optuna) nebo iterací (RandomSearch) |
| `metric` | string | Optimalizovaná metrika — `uar`, `accuracy`, `f1` |

Parametry které jsou v pipeline zadány jako **list** automaticky vstupují do searche. Skalár zůstává fixní.

---

## raw_aug

Modifikuje každý waveform před extrakcí příznaků (train i test). Sekce chybí = augmentace vypnuta.

| Parametr | Typ | Popis |
|---|---|---|
| `noise` | bool | Aditivní Gaussův bílý šum |
| `gain` | bool | Náhodné škálování amplitudy |
| `polarity` | bool | Inverze znaménka signálu |
| `time_crop` | bool | Náhodný ořez + zero-pad na původní délku |
| `pitch_shift` | bool | Posun výšky tónu, délka zachována — pomalé |
| `time_stretch` | bool | Změna rychlosti, výška zachována — pomalé |
| `reverb` | bool | Konvoluce se syntetickou impulsní odezvou místnosti |
| `codec` | bool | Simulace telefonního kodeku (mu-law) |
| `n_augmentations` | int / null | Počet náhodně vybraných technik na vzorek. `null` = všechny zapnuté |

---

## synthesizer

Generuje nové waveformy ze stávajících. Běží pouze na trénovacím foldu. Sekce chybí = syntéza vypnuta.

| Parametr | Typ | Popis |
|---|---|---|
| `n_samples` | int | Celkový počet nově generovaných řádků |
| `noise` | bool | Aditivní šum |
| `gain` | bool | Náhodné škálování amplitudy |
| `polarity` | bool | Inverze znaménka |
| `time_crop` | bool | Náhodný ořez |
| `pitch_shift` | bool | Posun výšky tónu — pomalé |
| `time_stretch` | bool | Změna rychlosti — pomalé |
| `reverb` | bool | Reverb |
| `codec` | bool | Simulace telefonního kodeku |
| `n_augmentations` | int / null | Počet náhodně vybraných technik na vzorek. `null` = všechny zapnuté |

---

## features

Každý extraktor je podsekce `features`. Přítomnost podsekce = extraktor aktivní, absence = přeskočen.

### features.mfcc

| Parametr | Typ | Popis |
|---|---|---|
| `n_mfcc` | int | Počet mel-frekvenčních kepstrálních koeficientů |
| `use_std` | bool | Zahrnout směrodatné odchylky koeficientů (zdvojuje počet příznaků) |

### features.timedomain

| Parametr | Typ | Popis |
|---|---|---|
| `use_f0` | bool | Střední hodnota základní frekvence |
| `use_stdf0` | bool | Směrodatná odchylka F0 |
| `use_diff_pitch` | bool | Relativní rozsah výšky tónu |
| `use_hnr` | bool | Harmonics-to-noise ratio |
| `use_jitter` | bool | Jitter (variabilita periody) |
| `use_shimmer` | bool | Shimmer (variabilita amplitudy) |
| `use_zcr` | bool | Zero crossing rate |
| `use_entropy` | bool | Entropie signálu |
| `use_skewness` | bool | Šikmost (skewness) |

### features.zcr

| Parametr | Typ | Popis |
|---|---|---|
| `use_mean` | bool | Průměr zero crossing rate |
| `use_std` | bool | Směrodatná odchylka zero crossing rate |

### features.spectral

| Parametr | Typ | Popis |
|---|---|---|
| `use_flatness` | bool | Spektrální placatost |
| `use_rolloff` | bool | Spektrální rolloff |
| `use_centroid` | bool | Spektrální centroid |
| `use_contrast` | bool | Spektrální kontrast (více pásem) |

### features.f0deriv

Robustní příznaky odvozené z F0, nezávislé na absolutní amplitudě.

| Parametr | Typ | Popis |
|---|---|---|
| `use_f0_iqr` | bool | Interkvartilové rozpětí F0 |
| `use_f0_range` | bool | Rozsah F0 (max − min) |
| `use_ppe` | bool | Pitch period entropy |
| `use_nan_rate` | bool | Podíl neznělých framů |
| `use_pitch_diff_std` | bool | Směrodatná odchylka relativní změny F0 mezi cykly |

### features.nonlinear

Nelineární dynamické příznaky. Signál je normalizován před výpočtem.

| Parametr | Typ | Popis |
|---|---|---|
| `use_rqa_rr` | bool | Recurrence rate (RQA) |
| `use_rqa_det` | bool | Determinism (RQA) |
| `use_rqa_lam` | bool | Laminarity (RQA) |
| `use_rpde` | bool | Recurrence period density entropy |
| `use_dfa` | bool | Detrended fluctuation analysis exponent |
| `use_sampen` | bool | Sample entropy |

### features.vocaltract

Modelování vokálního traktu jako cylindrických trubic pomocí lineární predikce.

| Parametr | Typ | Popis |
|---|---|---|
| `use_var` | bool | Variance oblasti trubic přes čas |
| `use_ratio_var` | bool | Normalizovaná variance |
| `use_kurtosis` | bool | Kurtosis oblasti trubic |

### features.opensmile

| Parametr | Typ | Popis |
|---|---|---|
| `feature_set` | string | Sada příznaků — `GeMAPS`, `ComParE_2016` |
| `feature_level` | string | Úroveň extrakce — `functionals`, `lld` |

### features.selector

Výběr příznaků po extrakci. Sekce chybí = selekce vypnuta.

| Parametr | Typ | Popis |
|---|---|---|
| `type` | string | Typ selektoru — `SelectKBest`, `SelectFromModel`, `RFECV` |
| `score_func` | string | Skórovací funkce pro SelectKBest — `mutual_info_classif`, `f_classif` |
| `k` | int | Počet příznaků pro SelectKBest |

---

## resampler

Vyvážení tříd na trénovacím foldu. Sekce chybí = resampling vypnut.

| Parametr | Typ | Popis |
|---|---|---|
| `type` | string | Typ resampleru — `SMOTESampler`, `VAESampler` |
| `sampling_strategy` | string | Strategie vyvážení — `auto`, `minority`, `not majority` |
| `artificial_only` | bool | Použít pouze syntetické vzorky bez originálních |

---

## clf

| Parametr | Typ | Popis |
|---|---|---|
| `type` | string | Klasifikátor — `RandomForestClassifier`, `KANClassifier` |
| `n_estimators` | int / list | Počet stromů. List = vstupuje do searche |
| `max_depth` | int / null / list | Maximální hloubka stromu. `null` = bez omezení |
| `min_samples_split` | int / list | Minimální počet vzorků pro rozdělení uzlu |
| `min_samples_leaf` | int / list | Minimální počet vzorků v listu |
| `max_features` | string / float / null / list | Počet příznaků pro hledání rozdělení — `sqrt`, `log2`, `null`, nebo číslo |
| `max_samples` | float / null / list | Podíl vzorků pro bootstrap. `null` = všechny |
| `criterion` | string / list | Kritérium rozdělení — `gini`, `entropy` |
| `class_weight` | string / null | Vyvážení tříd — `balanced`, `null` |

---

## cross_validation

| Parametr | Typ | Popis |
|---|---|---|
| `type` | string | Typ CV — `StratifiedGroupKFold` |
| `n_splits` | int | Počet foldů |
| `scoring_mode` | string | Režim scoringu — `multimetric`, `accuracy_only`, `uar_only` |
| `refit_metric` | string | Metrika pro výběr nejlepšího modelu — `uar`, `accuracy` |
| `n_repeats` | int | Počet opakování CV. `0` = vypnuto |

---

## visualization

| Parametr | Typ | Popis |
|---|---|---|
| `plot_uar_per_pathology` | bool | Graf UAR zvlášť pro každou patologii |
| `plot_feature_histograms` | bool | Histogramy distribucí příznaků |
