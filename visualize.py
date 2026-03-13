"""Visualization functions for experiment results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _uar_color(uar: float) -> str:
    if uar >= 0.75: return "#1D9E75"
    if uar >= 0.60: return "#378ADD"
    if uar >= 0.50: return "#EF9F27"
    return "#E24B4A"


def plot_uar_by_pathology(dfp, savepath=None, title="UAR per pathology"):
    d = dfp.sort_values("UAR", ascending=True).reset_index(drop=True)
    n = len(d)

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.45)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = [_uar_color(v) for v in d["UAR"]]
    bars = ax.barh(np.arange(n), d["UAR"].values, color=colors, height=0.6)

    # reference line at 0.5 (chance level)
    ax.axvline(0.5, color="#B4B2A9", linewidth=1, linestyle="--", zorder=0)

    # value + n labels to the right of each bar
    for i, (uar, n_pat) in enumerate(zip(d["UAR"], d["n_pathology"])):
        ax.text(
            min(0.97, uar + 0.015), i,
            f"{uar:.2f}  n={int(n_pat)}",
            va="center", ha="left",
            fontsize=10, color="#5F5E5A",
        )

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(d["pathology"].values, fontsize=11, color="#2C2C2A")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("UAR", fontsize=11, color="#5F5E5A")
    ax.set_title(title, fontsize=13, fontweight="normal", color="#2C2C2A", pad=12)

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.tick_params(axis="x", colors="#888780", labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#F1EFE8", zorder=0)
    ax.set_axisbelow(True)

    # color legend
    legend_items = [
        ("#1D9E75", "≥ 0.75"),
        ("#378ADD", "≥ 0.60"),
        ("#EF9F27", "≥ 0.50"),
        ("#E24B4A", "< 0.50"),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in legend_items]
    labels  = [l for _, l in legend_items]
    ax.legend(handles, labels, title="UAR", title_fontsize=9,
              fontsize=9, loc="lower right", framealpha=0.9,
              edgecolor="#D3D1C7")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_histograms(pipe, X, datasets, run_dir):
    """
    Plot histograms for each feature, with one distribution per dataset.
    Saves two sets: before scaling (after drop_sig/replace_nan) and after scaling.

    Parameters
    ----------
    pipe : fitted ImbPipeline
    X : raw input data
    datasets : np.ndarray of dataset labels per sample (e.g. "SVD", "AVFAD")
    run_dir : Path to save the plots
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    hist_dir_pre  = Path(run_dir) / "histograms_pre_scaling"
    hist_dir_post = Path(run_dir) / "histograms_post_scaling"
    hist_dir_pre.mkdir(exist_ok=True)
    hist_dir_post.mkdir(exist_ok=True)

    # split steps into pre-scaler and pre+scaler groups
    pre_steps, post_steps = [], []
    scaler_seen = False
    for name, step in pipe.steps:
        if name in ("selector", "feat_aug", "resampler", "clf"):
            break
        if not scaler_seen:
            pre_steps.append((name, step))
        else:
            post_steps.append((name, step))
        if name == "scaler":
            scaler_seen = True

    pre_pipe  = SkPipeline(pre_steps)
    post_pipe = SkPipeline(pre_steps + post_steps)

    X_pre  = pre_pipe.transform(X)
    X_post = post_pipe.transform(X)

    dataset_labels = np.unique(datasets)
    colors = plt.cm.tab10.colors

    for X_feat, hist_dir, label in [
        (X_pre,  hist_dir_pre,  "pre-scaling"),
        (X_post, hist_dir_post, "post-scaling"),
    ]:
        n_features = X_feat.shape[1]
        for i in range(n_features):
            fig, ax = plt.subplots(figsize=(7, 4))
            for j, ds in enumerate(dataset_labels):
                mask = datasets == ds
                ax.hist(
                    X_feat[mask, i],
                    bins=40, alpha=0.5,
                    color=colors[j % len(colors)],
                    label=ds, density=True,
                )
            ax.set_title(f"Feature {i} ({label})")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            fig.tight_layout()
            fig.savefig(hist_dir / f"feature_{i:03d}.png", dpi=120)
            plt.close(fig)


def run_visualizations(config, best_estimator, X, y, groups, pathologies, datasets, cv, run_dir):
    """
    Entry point called from main.py at the end of a run.
    Respects config switches:
        "plot_uar_per_pathology"   : bool (default True)
        "plot_feature_histograms"  : bool (default False)
    """
    from main import per_pathology_uar

    if config.get("plot_uar_per_pathology", True):
        dfp = per_pathology_uar(best_estimator, X, y, groups, pathologies, cv)

        dfp.to_csv(run_dir / "per_pathology_UAR_all.csv", index=False)
        plot_uar_by_pathology(dfp, run_dir / "uar_all.png")

        dfp_weak = dfp[dfp["UAR"] < dfp["UAR"].median()]
        dfp_weak.to_csv(run_dir / "per_pathology_UAR_weak.csv", index=False)
        plot_uar_by_pathology(dfp_weak, run_dir / "uar_weak.png")

    if config.get("plot_feature_histograms", False):
        plot_feature_histograms(best_estimator, X, datasets, run_dir)