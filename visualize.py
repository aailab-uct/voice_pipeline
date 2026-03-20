"""Visualization functions for experiment results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import clone
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline

def per_pathology_uar(best_estimator, X, y, groups, pathologies, cv, healthy_name="healthy"):
    """
    Evaluate classifier UAR separately for each pathology vs. healthy controls.

    For each fold, the estimator is re-fitted on the training split and used to
    predict the test split. Predictions are then aggregated across all folds and
    evaluated per pathology: each pathology is scored in a one-vs-healthy subset
    (samples belonging to that pathology + all healthy samples).

    Parameters
    ----------
    best_estimator : fitted sklearn estimator
        Estimator to evaluate. It is cloned and re-fitted for each fold,
        so the original object is not modified.
    X : np.ndarray
        Full feature matrix.
    y : np.ndarray
        Binary target labels (1 = pathological, 0 = healthy).
    groups : np.ndarray
        Group labels for cross-validation splitting (e.g. speaker IDs).
    pathologies : np.ndarray
        Pathology name for each sample (e.g. "Dysphonia", "healthy").
    cv : cross-validator
        Cross-validation splitter (e.g. StratifiedGroupKFold).
    healthy_name : str, optional
        Label used to identify healthy samples in ``pathologies``.
        Comparison is case-insensitive. Default: "healthy".

    Returns
    -------
    pd.DataFrame
        One row per pathology, sorted by UAR descending, with columns:
        pathology, UAR, TP, FN, FP, TN, n_pathology, n_healthy.
    """
    # --- Step 1: collect predictions across all CV folds ---
    # We need out-of-fold predictions for every sample, so we iterate over
    # all folds, refit the estimator on the training split, and collect
    # predictions on the held-out test split.
    y_true_all, y_pred_all, p_all = [], [], []

    for train, test in cv.split(X, y, groups):
        # Clone ensures each fold gets a fresh unfitted estimator —
        # without this, fit() would modify the shared best_estimator object.
        estimator = clone(best_estimator)
        estimator.fit(X[train], y[train])
        yp = estimator.predict(X[test])

        y_true_all.append(y[test])
        y_pred_all.append(yp)
        p_all.append(pathologies[test])

    # Concatenate fold results into single flat arrays for easy indexing
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    p_all = np.concatenate(p_all)

    # --- Step 2: compute UAR for each pathology vs. healthy ---
    # For each unique pathology we build a binary one-vs-healthy subset and
    # compute UAR (= mean per-class recall = balanced accuracy for binary case).
    # Healthy samples are always included as the negative class.
    rows = []
    for p in np.unique(p_all):
        # Skip the healthy class itself — it is the reference, not a target
        if p.lower() == healthy_name:
            continue

        # Build a boolean mask selecting only:
        #   - samples with this specific pathology (positive class)
        #   - all healthy samples (negative class)
        # This gives a binary classification subset for this pathology.
        m = (p_all == p) | (np.char.lower(p_all) == healthy_name)

        yt = y_true_all[m]
        yp = y_pred_all[m]

        # Compute confusion matrix entries using sklearn to avoid manual errors.
        # Rows = true class, columns = predicted class: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

        # UAR = unweighted average recall = mean of per-class recall scores.
        # Equivalent to balanced_accuracy_score but computed explicitly here
        # so we can also log individual recall values if needed in the future.
        r0, r1 = recall_score(yt, yp, labels=[0, 1], average=None, zero_division=0)
        uar = 0.5 * (r0 + r1)

        # Count how many samples of this pathology and how many healthy samples
        # are present in the subset (useful for interpreting UAR in context)
        n_pathology = int((p_all[m] == p).sum())
        n_healthy = int((np.char.lower(p_all[m]) == healthy_name).sum())

        rows.append((p, uar, tp, fn, fp, tn, n_pathology, n_healthy))

    df = pd.DataFrame(rows, columns=["pathology", "UAR", "TP", "FN", "FP", "TN", "n_pathology", "n_healthy"])
    return df.sort_values("UAR", ascending=False)

def _uar_color(uar: float) -> str:
    """
    Map a UAR value to a color for bar chart visualization.

    Color thresholds:
        >= 0.75  green  (good performance)
        >= 0.60  blue   (acceptable performance)
        >= 0.50  amber  (poor performance)
        <  0.50  red    (below chance)

    Parameters
    ----------
    uar : float
        UAR value in range [0, 1].

    Returns
    -------
    str
        Hex color string.
    """
    if uar >= 0.75:
        return "#1D9E75"
    if uar >= 0.60:
        return "#378ADD"
    if uar >= 0.50:
        return "#EF9F27"
    return "#E24B4A"

def plot_uar_by_pathology(dfp, savepath=None, title="UAR per pathology"):
    """
    Plot a horizontal bar chart of UAR scores per pathology.

    Bars are color-coded by UAR threshold (green/blue/amber/red).
    A dashed vertical line marks the chance level (UAR = 0.5).
    Each bar is annotated with the UAR value and sample count.

    Parameters
    ----------
    df_pathology : pd.DataFrame
        Output of ``per_pathology_uar()``. Must contain columns:
        ``pathology``, ``UAR``, ``n_pathology``.
    savepath : Path or str, optional
        If provided, the figure is saved to this path at 200 dpi.
    title : str, optional
        Plot title. Default: "UAR per pathology".
    """
    d = dfp.sort_values("UAR", ascending=True).reset_index(drop=True)  # what is dfp? bad naming imo
    n = len(d)

    fig, ax = plt.subplots(figsize=(7, max(2.5, n * 0.28)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = [_uar_color(v) for v in d["UAR"]]  # colors of what?
    ax.barh(np.arange(n), d["UAR"].values, color=colors, height=0.65)

    ax.axvline(0.5, color="#B4B2A9", linewidth=0.8, linestyle="--", zorder=0)

    for i, (uar, n_pat) in enumerate(zip(d["UAR"], d["n_pathology"])):
        ax.text(
            x = min(0.97, uar + 0.012),
            y = i,
            s = f"{uar:.2f}  n={int(n_pat)}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#5F5E5A",
        )

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(d["pathology"].values, fontsize=8, color="#2C2C2A")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("UAR", fontsize=9, color="#5F5E5A")
    ax.set_title(title, fontsize=10, fontweight="normal", color="#2C2C2A", pad=8)

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.tick_params(axis="x", colors="#888780", labelsize=8)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.xaxis.grid(True, color="#F1EFE8", zorder=0)
    ax.set_axisbelow(True)

    legend_items = [  # LOL vyber barev s takovouhle presnosti xD
        ("#1D9E75", "≥ 0.75"),
        ("#378ADD", "≥ 0.60"),
        ("#EF9F27", "≥ 0.50"),
        ("#E24B4A", "< 0.50"),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in legend_items]
    labels = [l for _, l in legend_items]
    ax.legend(
        handles,
        labels,
        title="UAR",
        title_fontsize=7,
        fontsize=7,
        loc="lower right",
        framealpha=0.9,
        edgecolor="#D3D1C7",
        handlelength=1,
        handleheight=0.8,
        borderpad=0.5,
        labelspacing=0.3,
    )

    fig.tight_layout(pad=0.5)
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
        The full fitted pipeline. Only steps up to and including the scaler are used.
    X : np.ndarray
        Raw input data.
    datasets : np.ndarray
        Dataset label for each sample (e.g. "SVD", "AVFAD").
    run_dir : Path
        Directory where histogram plots are saved.
    """
    hist_dir_pre = Path(run_dir) / "histograms_pre_scaling"
    hist_dir_post = Path(run_dir) / "histograms_post_scaling"
    hist_dir_pre.mkdir(exist_ok=True)
    hist_dir_post.mkdir(exist_ok=True)

    # Split pipeline steps into two groups:
    #   pre_steps  : everything before the scaler (raw feature space)
    #   post_steps : scaler itself
    # Steps after the scaler (selector, feat_aug, resampler, clf) are excluded
    # because they either require y (resampler), change feature count (selector),
    # or are not transformers (clf).
    STEPS_TO_EXCLUDE = {"selector", "feat_aug", "resampler", "clf"}
    pre_steps, post_steps = [], []
    scaler_seen = False
    for name, step in pipe.steps:
        if name in STEPS_TO_EXCLUDE:
            break
        if not scaler_seen:
            pre_steps.append((name, step))
        else:
            post_steps.append((name, step))
        if name == "scaler":
            scaler_seen = True

    # Use ImbPipeline for consistency with the rest of the project.
    # transform() does not trigger resampling (only fit_transform/fit does),
    # so it is safe to call without y here.
    pre_pipe = ImbPipeline(pre_steps)
    post_pipe = ImbPipeline(pre_steps + post_steps)

    X_pre = pre_pipe.transform(X)
    X_post = post_pipe.transform(X)

    dataset_labels = np.unique(datasets)

    # Build a color map with exactly as many colors as there are datasets.
    # Using a continuous colormap avoids the 10-color limit of tab10.
    cmap = plt.colormaps.get_cmap("tab10")
    colors = [cmap(i / max(len(dataset_labels) - 1, 1)) for i in range(len(dataset_labels))]

    for X_feat, hist_dir, label in [
        (X_pre, hist_dir_pre, "pre-scaling"),
        (X_post, hist_dir_post, "post-scaling"),
    ]:
        n_features = X_feat.shape[1]
        for i in range(n_features):
            fig, ax = plt.subplots(figsize=(7, 4))
            for j, ds in enumerate(dataset_labels):
                mask = datasets == ds
                ax.hist(
                    X_feat[mask, i],
                    bins=40,
                    alpha=0.5,
                    color=colors[j],
                    label=ds,
                    density=True,
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
    Generate and save visualizations after a completed grid search run.

    Called from main.py at the end of a run. Which plots are generated
    is controlled by config switches — unrecognized keys are silently ignored.

    Parameters
    ----------
    config : dict
        Experiment configuration. Recognized keys:
            ``plot_uar_per_pathology``  – bool, generate UAR per pathology plots (default: True)
            ``plot_feature_histograms`` – bool, generate feature histograms (default: False)
    best_estimator : fitted sklearn estimator
        Best estimator returned by GridSearchCV.
    X : np.ndarray
        Full feature matrix (unfitted).
    y : np.ndarray
        Target labels.
    groups : np.ndarray
        Group labels for cross-validation splitting.
    pathologies : np.ndarray
        Pathology label for each sample, used for per-pathology UAR breakdown.
    datasets : np.ndarray
        Dataset source label for each sample, used for feature histogram grouping.
    cv : cross-validator
        Cross-validation splitter (e.g. StratifiedGroupKFold).
    run_dir : Path
        Directory where output files are saved.

    Output files
    ------------
    per_pathology_UAR_all.csv  – UAR for all pathologies
    uar_all.png                – bar plot of UAR for all pathologies
    per_pathology_UAR_weak.csv – UAR for pathologies below median UAR
    uar_weak.png               – bar plot of weak pathologies only
    feature_histograms/        – per-feature histograms grouped by dataset (if enabled)
    """

    if config.get("plot_uar_per_pathology", True):
        dfp = per_pathology_uar(best_estimator, X, y, groups, pathologies, cv)

        dfp.to_csv(run_dir / "per_pathology_UAR_all.csv", index=False)
        plot_uar_by_pathology(dfp, run_dir / "uar_all.png")

        dfp_weak = dfp[dfp["UAR"] < dfp["UAR"].median()]
        dfp_weak.to_csv(run_dir / "per_pathology_UAR_weak.csv", index=False)
        plot_uar_by_pathology(dfp_weak, run_dir / "uar_weak.png")

    if config.get("plot_feature_histograms", False):
        plot_feature_histograms(best_estimator, X, datasets, run_dir)
