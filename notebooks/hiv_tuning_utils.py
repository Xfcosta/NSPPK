"""Reusable cohort selection and reporting helpers for the HIV tune notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def split_train_test(y, *, test_size=0.25, random_state=2026):
    """Return stratified full-data train and test row indices."""
    y = np.asarray(y)
    return train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def balanced_training_sample(y, train_indices, budget_per_class, *, random_state=2026):
    """Sample a balanced BO cohort from the full-training partition only."""
    y = np.asarray(y)
    train_indices = np.asarray(train_indices, dtype=int)
    class_indices = [train_indices[y[train_indices] == label] for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Both HIV classes must be represented in the training split.")
    if isinstance(budget_per_class, (bool, np.bool_)) or int(budget_per_class) < 1:
        raise ValueError("budget_per_class must be a positive integer.")
    effective_per_class = min(int(budget_per_class), *(len(indices) for indices in class_indices))
    rng = np.random.default_rng(random_state)
    sampled_indices = np.concatenate([
        rng.choice(indices, size=effective_per_class, replace=False)
        for indices in class_indices
    ])
    rng.shuffle(sampled_indices)
    return sampled_indices, effective_per_class


def tuning_result_frame(result):
    """Convert an OptimizeResult from NSPPK .tune() to a trial table."""
    parameter_names = list(result.search_space)
    fold_scores = np.asarray(result.fold_scores, dtype=float)
    rows = []
    for trial_index, (values, score) in enumerate(
        zip(result.x_iters, result.cv_scores), start=1
    ):
        folds = fold_scores[trial_index - 1]
        rows.append({
            **dict(zip(parameter_names, values)),
            "trial": trial_index,
            "score": float(score),
            "score_std": float(np.std(folds, ddof=1)) if len(folds) > 1 else 0.0,
            "fold_scores": folds.tolist(),
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def plot_tuning_progress(results, *, score_column="score", title="Bayesian optimization progress"):
    """Plot trial scores in execution order."""
    progress = results.sort_values("trial")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(progress["trial"], progress[score_column], marker="o", alpha=0.8)
    ax.axhline(results[score_column].max(), color="tab:red", linestyle="--", label="Best observed")
    ax.set(xlabel="BO trial", ylabel=score_column, title=title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def heldout_metrics(y_true, probabilities, predictions):
    """Compute held-out binary classification metrics."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "precision_at_0.5": precision_score(y_true, predictions, zero_division=0),
        "recall_at_0.5": recall_score(y_true, predictions, zero_division=0),
    }


def plot_heldout_report(
    y_true,
    probabilities,
    predictions,
    *,
    best_params,
    nbits,
    classifier_label="Random forest, 300 trees",
):
    """Create a compact score bar chart and confusion/ROC/PR diagnostics."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)
    metrics = heldout_metrics(y_true, probabilities, predictions)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    cm_fraction = cm / cm.sum(axis=1, keepdims=True)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[0.85, 1.4])
    ax_metrics = fig.add_subplot(gs[0, :])
    metric_names = ["Accuracy", "Balanced accuracy", "Macro F1", "ROC AUC", "Average precision"]
    metric_values = [
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["roc_auc"],
        metrics["average_precision"],
    ]
    bars = ax_metrics.bar(
        metric_names,
        metric_values,
        color=["#6b7280", "#2563eb", "#0f766e", "#7c3aed", "#ea580c"],
        width=0.68,
    )
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Held-out test metrics")
    ax_metrics.grid(axis="y", alpha=0.25)
    ax_metrics.bar_label(bars, fmt="%.3f", padding=3)

    ax_cm = fig.add_subplot(gs[1, 0])
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set(title="Confusion (counts)", xlabel="Predicted class", ylabel="True class", xticks=[0, 1], yticks=[0, 1])
    for row in range(2):
        for col in range(2):
            ax_cm.text(
                col, row, f"{cm[row, col]}", ha="center", va="center",
                color="white" if cm[row, col] > cm.max() * 0.55 else "black",
            )
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_fraction = fig.add_subplot(gs[1, 1])
    im_fraction = ax_fraction.imshow(cm_fraction, cmap="Blues", vmin=0, vmax=1)
    ax_fraction.set(
        title="Confusion (row fractions)", xlabel="Predicted class", ylabel="True class",
        xticks=[0, 1], yticks=[0, 1],
    )
    for row in range(2):
        for col in range(2):
            ax_fraction.text(
                col, row, f"{cm_fraction[row, col]:.3f}", ha="center", va="center",
                color="white" if cm_fraction[row, col] > 0.55 else "black",
            )
    fig.colorbar(im_fraction, ax=ax_fraction, fraction=0.046, pad=0.04)

    ax_roc = fig.add_subplot(gs[1, 2])
    ax_roc.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}", color="#2563eb")
    ax_roc.plot([0, 1], [0, 1], ":", color="black", label="Random baseline")
    ax_roc.set(title="ROC", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.grid(alpha=0.25)
    ax_roc.legend(frameon=False, loc="lower right")

    ax_pr = fig.add_subplot(gs[1, 3])
    ax_pr.plot(recall, precision, label=f"Average precision = {metrics['average_precision']:.3f}", color="#2563eb")
    ax_pr.axhline(y_true.mean(), linestyle=":", color="black", label=f"Random baseline = {y_true.mean():.3f}")
    ax_pr.set(title="Precision–recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.grid(alpha=0.25)
    ax_pr.legend(frameon=False, loc="upper right")

    fig.suptitle("HIV: held-out test model performance", fontsize=16)
    subtitle = f"{classifier_label} | NSPPK {best_params} | nbits={nbits}"
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=10)
    return metrics, cm, fig
