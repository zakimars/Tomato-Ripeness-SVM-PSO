"""Train and compare K-Means, default SVM and SVM-PSO; save the best model.

Usage:
    python -m tomato.train --data data/tomato_rgbc.csv
    python -m tomato.train --data data/tomato_rgbc.csv --split group   # needs a fruit_id column
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from tomato import CLASSES, FEATURES  # noqa: E402
from tomato.features import load_readings, preprocess, xy  # noqa: E402
from tomato.models import SEARCH_SPACES, KMeansClassifier, decode, make_svm  # noqa: E402
from tomato.pso import pso_maximize  # noqa: E402


def metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f}


def bootstrap_std(y_true, y_pred, n: int, seed: int) -> dict:
    """Std of each metric over bootstrap resamples of the test set (thesis Table 4.10)."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        samples.append(metrics(y_true[idx], y_pred[idx]))
    return {k: float(np.std([s[k] for s in samples])) for k in samples[0]}


def split(df, how: str, test_size: float, seed: int):
    if how == "stratified":
        tr, te = train_test_split(df.index, test_size=test_size, stratify=df["y"], random_state=seed)
        return df.loc[tr].reset_index(drop=True), df.loc[te].reset_index(drop=True)
    if "fruit_id" not in df.columns:
        raise SystemExit("--split group needs a 'fruit_id' column so readings of one tomato stay together.")
    n_splits = max(2, round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(df, df["y"], groups=df["fruit_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def cv_splitter(train_df, how: str, seed: int):
    if how == "group":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return list(cv.split(train_df, train_df["y"], groups=train_df["fruit_id"]))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(train_df, train_df["y"]))


def plot_confusions(results: dict, y_test, out: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        cm = confusion_matrix(y_test, r["y_pred"], labels=range(len(CLASSES)))
        ConfusionMatrixDisplay(cm, display_labels=["unripe", "partial", "ripe"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\naccuracy {r['metrics']['accuracy']:.2%}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_convergence(histories: dict, out: Path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for kernel, h in histories.items():
        ax.plot(h, label=kernel)
    ax.set_xlabel("iteration")
    ax.set_ylabel("best 5-fold CV accuracy")
    ax.set_title("PSO convergence per kernel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/tomato_rgbc.csv")
    ap.add_argument("--split", choices=["stratified", "group"], default="stratified")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--kernels", nargs="+", default=["rbf", "sigmoid", "poly"], choices=list(SEARCH_SPACES))
    ap.add_argument("--particles", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default="results")
    ap.add_argument("--model-out", default="models/svm_pso.joblib")
    args = ap.parse_args(argv)

    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    df = preprocess(load_readings(args.data))
    print(f"{len(df)} valid readings | per class: {df['label'].value_counts().to_dict()}")
    train_df, test_df = split(df, args.split, args.test_size, args.seed)
    X_tr, y_tr = xy(train_df)
    X_te, y_te = xy(test_df)
    print(f"split={args.split}: {len(train_df)} train / {len(test_df)} test")

    folds = cv_splitter(train_df, args.split, args.seed)
    results: dict[str, dict] = {}

    km = KMeansClassifier(seed=args.seed).fit(X_tr, y_tr)
    results["K-Means"] = {"y_pred": km.predict(X_te)}

    tuned: dict[str, dict] = {}
    histories: dict[str, list[float]] = {}
    for kernel in args.kernels:
        default = make_svm(kernel).fit(X_tr, y_tr)
        results[f"SVM default ({kernel})"] = {"y_pred": default.predict(X_te)}

        bounds = np.array([b for _, b in SEARCH_SPACES[kernel]])

        def fitness(position, kernel=kernel):
            model = make_svm(kernel, **decode(kernel, position))
            return float(cross_val_score(model, X_tr, y_tr, cv=folds, scoring="accuracy").mean())

        print(f"PSO for {kernel} kernel ...")
        res = pso_maximize(fitness, bounds[:, 0], bounds[:, 1], args.particles, args.iters, seed=args.seed, verbose=True)
        params = decode(kernel, res.best_position)
        model = make_svm(kernel, **params).fit(X_tr, y_tr)
        tuned[kernel] = {"params": params, "cv_fitness": res.best_fitness, "model": model}
        histories[kernel] = res.history
        results[f"SVM-PSO ({kernel})"] = {"y_pred": model.predict(X_te)}

    for r in results.values():
        r["metrics"] = metrics(y_te, r["y_pred"])

    # Thesis rule: pick the kernel with the best CV fitness (not the best test score).
    best_kernel = max(tuned, key=lambda k: tuned[k]["cv_fitness"])
    best = tuned[best_kernel]
    headline = {
        "K-Means": results["K-Means"],
        "SVM (default)": results.get(f"SVM default ({best_kernel})"),
        f"SVM-PSO ({best_kernel})": results[f"SVM-PSO ({best_kernel})"],
    }
    for r in headline.values():
        r["bootstrap_std"] = bootstrap_std(y_te, r["y_pred"], args.bootstrap, args.seed)

    report = {
        "split": args.split,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "best_kernel": best_kernel,
        "best_params": best["params"],
        "pso": {k: {"params": v["params"], "cv_fitness": v["cv_fitness"]} for k, v in tuned.items()},
        "test_metrics": {k: v["metrics"] for k, v in results.items()},
        "bootstrap_std": {k: v["bootstrap_std"] for k, v in headline.items()},
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    lines = ["| Model | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for name, r in results.items():
        m = r["metrics"]
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_confusions(headline, y_te, out / "confusion_matrices.png")
    plot_convergence(histories, out / "pso_convergence.png")

    joblib.dump(
        {"model": best["model"], "kernel": best_kernel, "params": best["params"], "features": FEATURES, "classes": CLASSES},
        args.model_out,
    )
    print(f"\nBest kernel: {best_kernel} {best['params']}  (CV fitness {best['cv_fitness']:.4f})")
    print(f"Saved model -> {args.model_out}, reports -> {out}/")


if __name__ == "__main__":
    main()
