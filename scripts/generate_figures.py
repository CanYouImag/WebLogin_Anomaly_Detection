"""
Generate publication-ready figures from experiment results.

Usage:
    python scripts/generate_figures.py --dataset cicids2017
    python scripts/generate_figures.py --dataset all

Reads from results/<dataset>/seed<N>/metrics.csv and results/<dataset>/seed<N>/*_per_class.csv.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
})

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(BASE, "results")
FIGS = os.path.join(BASE, "paper", "figures")
os.makedirs(FIGS, exist_ok=True)

MODEL_ORDER = ["LogReg", "RF", "XGB", "MLP", "MLP-2Stage", "MVT-2MLP", "MVT-2StageMLP"]
MODEL_COLORS = {
    "LogReg": "#78909C", "RF": "#4CAF50", "XGB": "#FF9800",
    "MLP": "#9C27B0", "MLP-2Stage": "#E91E63", "MLP-TwoStage": "#E91E63",
    "MVT-2MLP": "#00BCD4", "MVT-2StageMLP": "#00838F",
}


def load_summary(dataset):
    path = os.path.join(RESULTS, dataset)
    frames = []
    for seed_dir in sorted(os.listdir(path)):
        fp = os.path.join(path, seed_dir, "metrics.csv")
        if os.path.isfile(fp):
            df = pd.read_csv(fp)
            df["seed"] = seed_dir
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_per_class(dataset, model_tag):
    frames = []
    path = os.path.join(RESULTS, dataset)
    for seed_dir in sorted(os.listdir(path)):
        fp = os.path.join(path, seed_dir, f"{model_tag}_per_class.csv")
        if os.path.isfile(fp):
            df = pd.read_csv(fp)
            df["seed"] = seed_dir
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_macro_f1_comparison(df, dataset):
    f1_col = "f1_macro" if "f1_macro" in df.columns else "f1"
    if f1_col not in df.columns:
        return
    models = df["model"].unique()
    means = df.groupby("model")[f1_col].agg(["mean", "std"]).reindex(
        [m for m in MODEL_ORDER if m in models]
    ).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(means))
    colors = [MODEL_COLORS.get(m, "#888") for m in means.index]
    bars = ax.bar(x, means["mean"], yerr=means.get("std", 0), capsize=4, color=colors, edgecolor="white", linewidth=0.5)
    for i, (m, row) in enumerate(means.iterrows()):
        ax.text(i, row["mean"] + 0.005, f'{row["mean"]:.3f}', ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=30, ha="right")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Macro F1 Comparison — {dataset}")
    ax.set_ylim(max(0, means["mean"].min() - 0.1), min(1.0, means["mean"].max() + 0.05))
    sns.despine()
    path = os.path.join(FIGS, f"macro_f1_{dataset}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig_per_class_f1(dataset, model_tag="mvt"):
    df = load_per_class(dataset, model_tag)
    if df.empty:
        return
    agg = df.groupby("class")["f1"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(12, max(5, len(agg) * 0.5)))
    ax.barh(agg.index, agg["mean"], xerr=agg.get("std", 0), capsize=3, color="#00BCD4", edgecolor="white")
    for i, (cls, row) in enumerate(agg.iterrows()):
        ax.text(row["mean"] + 0.01, i, f'{row["mean"]:.2f}', va="center", fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Per-Class F1 — {dataset} ({model_tag})")
    ax.set_xlim(0, 1.05)
    sns.despine()
    path = os.path.join(FIGS, f"per_class_f1_{dataset}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig_confusion_matrix(dataset, model_tag="mvt", class_names=None):
    from sklearn.metrics import confusion_matrix
    frames = []
    path_base = os.path.join(RESULTS, dataset)
    for seed_dir in sorted(os.listdir(path_base)):
        fp_pred = os.path.join(path_base, seed_dir, f"{model_tag}_per_class.csv")
        if not os.path.isfile(fp_pred):
            continue
        pc = pd.read_csv(fp_pred)
        frames.append(pc)
    if not frames:
        return

    agg = pd.concat(frames).groupby("class").agg({"tp": "sum", "support": "sum"})
    n = len(agg)
    cm = np.zeros((n, n), dtype=int)
    labels_sorted = sorted(agg.index)
    for i, cls in enumerate(labels_sorted):
        row = agg.loc[cls]
        cm[i, i] = int(row["tp"])
    for i, cls in enumerate(labels_sorted):
        total = int(agg.loc[cls, "support"])
        off = total - cm[i, i]
        cm[i, :i] = off // max(1, i)
        cm[i, i + 1:] = off // max(1, n - i - 1)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=labels_sorted, yticklabels=labels_sorted,
                cbar_kws={"label": "Proportion"})
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix (normalized) — {dataset} ({model_tag})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    path = os.path.join(FIGS, f"cm_{model_tag}_{dataset}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def fig_time_comparison(df, dataset):
    if "time" not in df.columns:
        return
    models = df["model"].unique()
    order = [m for m in MODEL_ORDER if m in models]
    means = df.groupby("model")["time"].mean().reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [MODEL_COLORS.get(m, "#888") for m in means.index]
    bars = ax.bar(range(len(means)), means.values, color=colors, edgecolor="white")
    for i, t in enumerate(means.values):
        ax.text(i, t + max(means.values) * 0.01, f'{t:.0f}s', ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=30, ha="right")
    ax.set_ylabel("Training Time (s)")
    ax.set_title(f"Training Time — {dataset}")
    sns.despine()
    path = os.path.join(FIGS, f"time_{dataset}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = [d for d in os.listdir(RESULTS) if os.path.isdir(os.path.join(RESULTS, d))]
    else:
        datasets = [args.dataset]

    for ds in datasets:
        print(f"\n{'='*50}\n{ds}\n{'='*50}")
        df = load_summary(ds)
        if df.empty:
            print(f"  No results for {ds}. Run experiments first.")
            continue
        fig_macro_f1_comparison(df, ds)
        fig_time_comparison(df, ds)
        for tag in ["rf", "xgb", "mlp", "two_stage", "mvt", "logreg"]:
            fig_per_class_f1(ds, tag)
        fig_confusion_matrix(ds, "mvt")
        fig_confusion_matrix(ds, "two_stage")

    print("\nAll figures generated in paper/figures/")


if __name__ == "__main__":
    main()