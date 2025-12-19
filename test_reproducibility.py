import pandas as pd
from pathlib import Path
from main import main
from importlib import import_module

for i in range(5):
    parsed_config = import_module("config").config
    main(parsed_config)

results = Path("results").glob("*.csv")
dfs = []
for result_file in results:
    df = pd.read_csv(result_file)
    df = df.drop(columns=["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"])
    dfs.append(df)

final_df = pd.concat(dfs)
# test if all rows with the same param combination have the same mean_test_uar
grouped = final_df.groupby(
    ["param_raw_aug__enabled", "param_feat_aug__enabled", "param_feat_aug__std_factor", "param_clf__n_neighbors"]
)
for stat in ["mean_test_uar", "std_test_uar", "mean_test_TP", "mean_test_TN"]:
    for name, group in grouped:
        uar_values = group[stat].unique()
        assert len(uar_values) == 1, f"Different UAR values for params {name}: {uar_values}"
        print(f"Params {name} have consistent {stat}: {uar_values[0]}")
