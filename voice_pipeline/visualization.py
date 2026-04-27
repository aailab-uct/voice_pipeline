"""Visualization helpers for experiment results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline


STEPS_TO_EXCLUDE = {"selector", "feat_aug", "resampler", "clf"}  # TODO what this does?


def per_pathology_recall(y_true, y_pred, pathologies, healthy_name="healthy", min_samples=3):
    """Compute recall for each pathology against healthy controls.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels (1 = pathological, 0 = healthy).
    y_pred : numpy.ndarray
        Predicted binary labels from the classifier.
    pathologies : numpy.ndarray
        Pathology name for each sample.
    healthy_name : str, default="healthy"
        Label used to identify healthy samples. Comparison is
        case-insensitive.
    min_samples : int, default=3
        Minimum number of pathology samples required for inclusion.

    Returns
    -------
    pandas.DataFrame
        One row per pathology, sorted by Recall descending, with columns:
        ``pathology``, ``Recall``, ``TP``, ``FN``, ``FP``, ``TN``,
        ``n_pathology``, and ``n_healthy``.
    """
    rows = []
    for p in np.unique(pathologies):
        if p.lower() == healthy_name.lower():
            continue

        # One-vs-healthy subset
        m = (pathologies == p) | (np.char.lower(pathologies.astype(str)) == healthy_name.lower())

        n_pathology = int((pathologies[m] == p).sum())
        if n_pathology < min_samples:
            continue

        yt = y_true[m]
        yp = y_pred[m]

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

        recall = recall_score(yt, yp, zero_division=0)
        n_healthy = int((np.char.lower(pathologies[m].astype(str)) == healthy_name.lower()).sum())

        rows.append((p, recall, tp, fn, fp, tn, n_pathology, n_healthy))

    df = pd.DataFrame(rows, columns=["pathology", "Recall", "TP", "FN", "FP", "TN", "n_pathology", "n_healthy"])
    return df.sort_values("Recall", ascending=False)


def _recall_color(recall: float) -> str:
    """Map a recall value to a bar color.

    Parameters
    ----------
    recall : float
        Recall score in the interval [0, 1].

    Returns
    -------
    str
        Hex color string.
    """
    if recall >= 0.75:
        return "#1D9E75"
    if recall >= 0.60:
        return "#378ADD"
    if recall >= 0.50:
        return "#EF9F27"
    return "#E24B4A"


def plot_recall_by_pathology(df, savepath=None, title="Recall per pathology"):
    """Plot recall scores by pathology.

    Bars are color-coded by Recall threshold (green/blue/amber/red).
    A dashed vertical line marks the chance level (Recall = 0.5).
    Each bar is annotated with the Recall value and sample count.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``per_pathology_recall()``. Must contain columns:
        ``pathology``, ``Recall``, ``n_pathology``.
    savepath : pathlib.Path, str, or None, default=None
        If provided, the figure is saved to this path at 200 dpi.
    title : str, default="Recall per pathology"
        Plot title.
    """
    d = df.sort_values("Recall", ascending=True).reset_index(drop=True)
    n = len(d)

    fig, ax = plt.subplots(figsize=(7, max(2.5, n * 0.28)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = [_recall_color(v) for v in d["Recall"]]
    ax.barh(np.arange(n), d["Recall"].values, color=colors, height=0.65)

    ax.axvline(0.5, color="#B4B2A9", linewidth=0.8, linestyle="--", zorder=0)

    for i, (recall, n_pat) in enumerate(zip(d["Recall"], d["n_pathology"])):
        ax.text(
            x=min(0.97, recall + 0.012),
            y=i,
            s=f"{recall:.2f}  n={int(n_pat)}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#5F5E5A",
        )

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(d["pathology"].values, fontsize=8, color="#2C2C2A")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Recall", fontsize=9, color="#5F5E5A")
    ax.set_title(title, fontsize=10, fontweight="normal", color="#2C2C2A", pad=8)

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D3D1C7")
    ax.tick_params(axis="x", colors="#888780", labelsize=8)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.xaxis.grid(True, color="#F1EFE8", zorder=0)
    ax.set_axisbelow(True)

    legend_items = [
        ("#1D9E75", ">= 0.75"),
        ("#378ADD", ">= 0.60"),
        ("#EF9F27", ">= 0.50"),
        ("#E24B4A", "< 0.50"),
    ]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in legend_items]
    labels = [label for _, label in legend_items]
    ax.legend(
        handles,
        labels,
        title="Recall",
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


def plot_roc_curve(best_estimator, X, y, groups, cv, run_dir):
    """Plot the mean ROC curve across CV folds.

    Parameters
    ----------
    best_estimator : estimator
        Estimator to clone and refit in each fold.
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target labels.
    groups : numpy.ndarray
        Group labels for cross-validation.
    cv : cross-validator
        Cross-validation splitter.
    run_dir : pathlib.Path
        Directory where ``roc_curve.png`` and ``roc_data.csv`` are saved.
    """
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y, groups):
        est = clone(best_estimator)
        est.fit(X[train], y[train])
        y_proba = est.predict_proba(X[test])[:, 1]

        fpr, tpr, _ = roc_curve(y[test], y_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(roc_auc_score(y[test], y_proba))

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_fpr, mean_tpr, color="#1D9E75", label=f"Mean ROC (AUC = {mean_auc:.3f} +/- {std_auc:.3f})", lw=2)
    ax.fill_between(
        mean_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        alpha=0.15,
        color="#1D9E75",
        label="+/- 1 std",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    ax.set_xlabel("1 - Specificity (FPR)", fontsize=10)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=10)
    ax.set_title("ROC Curve", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run_dir / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        {
            "fpr": mean_fpr,
            "tpr_mean": mean_tpr,
            "tpr_std": std_tpr,
        }
    ).to_csv(run_dir / "roc_data.csv", index=False)


def plot_feature_histograms(pipe, X, datasets, run_dir):
    """Plot feature histograms grouped by dataset.

    Parameters
    ----------
    pipe : imblearn.pipeline.Pipeline
        Fitted pipeline. Only preprocessing steps through the scaler are used.
    X : numpy.ndarray
        Raw input data.
    datasets : numpy.ndarray
        Dataset label for each sample.
    run_dir : pathlib.Path
        Directory where histogram plots are saved.
    """
    hist_dir_pre = Path(run_dir) / "histograms_pre_scaling"
    hist_dir_post = Path(run_dir) / "histograms_post_scaling"
    hist_dir_pre.mkdir(exist_ok=True)
    hist_dir_post.mkdir(exist_ok=True)

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

    pre_pipe = ImbPipeline(pre_steps)
    post_pipe = ImbPipeline(pre_steps + post_steps)

    X_pre = pre_pipe.transform(X)
    X_post = post_pipe.transform(X)

    dataset_labels = np.unique(datasets)

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
    """Generate visual outputs for a completed grid search.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    best_estimator : estimator
        Best estimator returned by ``GridSearchCV``.
    X : numpy.ndarray
        Full feature matrix.
    y : numpy.ndarray
        Target labels.
    groups : numpy.ndarray
        Group labels for cross-validation splitting.
    pathologies : numpy.ndarray
        Pathology label for each sample.
    datasets : numpy.ndarray
        Dataset source label for each sample, used for feature histogram grouping.
    cv : cross-validator
        Cross-validation splitter used for ROC curve and feature histograms.
    run_dir : pathlib.Path
        Directory where output files are saved.
    """
    if config.get("plot_recall_per_pathology", True):
        min_samples = config.get("min_pathology_samples", 3)

        # Classify the full dataset in one pass with the fitted estimator
        y_pred = best_estimator.predict(X)

        df = per_pathology_recall(y, y_pred, pathologies, min_samples=min_samples)

        df.to_csv(run_dir / "per_pathology_recall_all.csv", index=False)
        plot_recall_by_pathology(df, run_dir / "recall_all.png")

        df_weak = df[df["Recall"] < df["Recall"].median()]
        df_weak.to_csv(run_dir / "per_pathology_recall_weak.csv", index=False)
        plot_recall_by_pathology(df_weak, run_dir / "recall_weak.png", title="Recall per pathology (weak)")

    if config.get("plot_roc_curve", True):
        if not hasattr(best_estimator[-1], "predict_proba"):
            print("Classifier does not support predict_proba, skipping ROC plot.")
        else:
            plot_roc_curve(best_estimator, X, y, groups, cv, run_dir)

    if config.get("plot_feature_histograms", False):
        plot_feature_histograms(best_estimator, X, datasets, run_dir)
