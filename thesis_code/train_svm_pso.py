"""Original thesis training program (Lampiran 3), restored from the PDF listing.

Changes from the printed listing — only what was needed to make it run:
  * `import mpl_toolkits.mplot3d import Axes3D` -> `from mpl_toolkits.mplot3d import Axes3D`
  * run_pso() now records the best fitness per iteration (`histories`, used by Figure 4.2)
  * added bootstrap_std() (used by Figure 4.3 / Table 4.10)
  * defined y_pred_default_terbaik / y_pred_terbaik (used by Figures 4.3 and 4.4)
  * command-line options for the data file and iterations; defaults are the thesis values
  * figures are saved without opening windows (--show to display them)

Note: as in the thesis code, the final kernel is chosen by *test* accuracy (ties -> RBF).
The refactored pipeline in src/tomato chooses by cross-validation fitness instead.

    python thesis_code/train_svm_pso.py --data dataset_tomat.xlsx
"""

import argparse

import joblib
import matplotlib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", default="dataset_tomat.xlsx", help=".xlsx (sheet 'Data Gabungan') or .csv")
parser.add_argument("--sheet", default="Data Gabungan")
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--particles", type=int, default=20)
parser.add_argument("--model-out", default="model_svm_pso.pkl")
parser.add_argument("--show", action="store_true", help="show figures in a window")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers the 3D projection)
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

NAMA_FILE = args.data
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

if NAMA_FILE.endswith(".csv"):
    df = pd.read_csv(NAMA_FILE)
else:
    df = pd.read_excel(NAMA_FILE, sheet_name=args.sheet)
features_norm = ["R_norm", "G_norm", "B_norm"]
label_map = {"Tidak Matang": 0, "Setengah Matang": 1, "Matang": 2}
X = df[features_norm].values
y = np.array([label_map[l] for l in df["Label"].values])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def metrik(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def bootstrap_std(y_true, y_pred, n_boot=1000):
    """Std of each metric over bootstrap resamples of the test set (Table 4.10)."""
    rng = np.random.RandomState(RANDOM_STATE)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        m = metrik(y_true[idx], y_pred[idx])
        scores.append([m["Accuracy"], m["Precision"], m["Recall"], m["F1-Score"]])
    return np.array(scores).std(axis=0)


# K-Means warna sebagai pembanding (unsupervised)
km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
clusters_train = km.fit_predict(X_train)
cluster_to_label = {c: int(np.bincount(y_train[clusters_train == c]).argmax()) for c in range(3)}
y_pred_km = np.array([cluster_to_label[c] for c in km.predict(X_test)])
hasil_km = metrik(y_test, y_pred_km)

# K-Means berat tomat untuk penentuan threshold validasi load cell (n=60)
berat_data = {
    "Tidak Matang": [30, 32, 33, 35, 35, 36, 37, 38, 38, 40, 42, 49, 69, 61, 63, 48, 62, 49, 67, 64, 71, 46, 65],
    "Setengah Matang": [43, 44, 45, 46, 47, 48, 50, 50, 51, 52, 63, 42, 51, 61, 72, 58, 47, 46, 50, 64, 40, 44, 56, 44, 49, 58],
    "Matang": [52, 53, 56, 56, 57, 58, 59, 61, 63, 69, 65],
}
berat_all, label_asli_berat = [], []
for kelas, nilai in berat_data.items():
    berat_all.extend(nilai)
    label_asli_berat.extend([kelas] * len(nilai))
berat_arr = np.array(berat_all).reshape(-1, 1)

km_berat = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
km_berat.fit(berat_arr)
centers = sorted(km_berat.cluster_centers_.flatten())
urutan = {0: "Tidak Matang", 1: "Setengah Matang", 2: "Matang"}
order_map = {old: new for new, old in enumerate(np.argsort(km_berat.cluster_centers_.flatten()))}
cluster_pred_berat = [urutan[order_map[l]] for l in km_berat.labels_]

batas_01 = (centers[0] + centers[1]) / 2
batas_12 = (centers[1] + centers[2]) / 2
sil_berat = silhouette_score(berat_arr, km_berat.labels_)
akurasi_berat = sum(a == b for a, b in zip(label_asli_berat, cluster_pred_berat)) / len(berat_all)
cm_berat = confusion_matrix(label_asli_berat, cluster_pred_berat, labels=["Tidak Matang", "Setengah Matang", "Matang"])

# SVM parameter default
kernels = ["poly", "sigmoid", "rbf"]
hasil_default = {}
y_pred_default = {}
for kernel in kernels:
    model = SVC(kernel=kernel, random_state=RANDOM_STATE).fit(X_train, y_train)
    y_pred_default[kernel] = model.predict(X_test)
    hasil_default[kernel] = metrik(y_test, y_pred_default[kernel])

# Konfigurasi PSO
KERNEL_BOUNDS = {
    "poly": {"C": (0.01, 1000), "gamma": (0.0001, 10), "d": (1, 5), "r": (-1, 1)},
    "sigmoid": {"C": (0.01, 1000), "gamma": (0.0001, 10), "r": (-1, 1)},
    "rbf": {"C": (0.01, 1000), "gamma": (0.0001, 10)},
}
KERNEL_PARAMS = {"poly": ["C", "gamma", "d", "r"], "sigmoid": ["C", "gamma", "r"], "rbf": ["C", "gamma"]}
N_PARTICLES, N_ITERATIONS = args.particles, args.iterations
W_MAX, W_MIN, C1, C2 = 0.9, 0.4, 2.0, 2.0


def fitness(params, kernel):
    try:
        kw = {"kernel": kernel, "C": params["C"], "random_state": RANDOM_STATE}
        if "gamma" in params:
            kw["gamma"] = params["gamma"]
        if "d" in params:
            kw["degree"] = int(round(params["d"]))
        if "r" in params:
            kw["coef0"] = params["r"]
        return cross_val_score(SVC(**kw), X_train, y_train, cv=cv, scoring="accuracy").mean()
    except Exception:
        return 0.0


def to_params(vec, kernel):
    return {p: vec[i] for i, p in enumerate(KERNEL_PARAMS[kernel])}


def clip(vec, kernel):
    out = vec.copy()
    for i, p in enumerate(KERNEL_PARAMS[kernel]):
        low, high = KERNEL_BOUNDS[kernel][p]
        out[i] = np.clip(out[i], low, high)
    return out


def run_pso(kernel):
    np.random.seed(RANDOM_STATE)
    names = KERNEL_PARAMS[kernel]
    n = len(names)
    pos = np.zeros((N_PARTICLES, n))
    vel = np.zeros((N_PARTICLES, n))
    for i, p in enumerate(names):
        low, high = KERNEL_BOUNDS[kernel][p]
        pos[:, i] = np.random.uniform(low, high, N_PARTICLES)
        vel[:, i] = np.random.uniform(-(high - low) * 0.1, (high - low) * 0.1, N_PARTICLES)
    fit = np.array([fitness(to_params(pos[i], kernel), kernel) for i in range(N_PARTICLES)])
    pbest_pos, pbest_fit = pos.copy(), fit.copy()
    g = np.argmax(pbest_fit)
    gbest_pos, gbest_fit = pbest_pos[g].copy(), pbest_fit[g]
    history = [gbest_fit]
    for t in range(1, N_ITERATIONS + 1):
        w = W_MAX - (W_MAX - W_MIN) * t / N_ITERATIONS
        for i in range(N_PARTICLES):
            r1, r2 = np.random.random(n), np.random.random(n)
            vel[i] = w * vel[i] + C1 * r1 * (pbest_pos[i] - pos[i]) + C2 * r2 * (gbest_pos - pos[i])
            pos[i] = clip(pos[i] + vel[i], kernel)
            f = fitness(to_params(pos[i], kernel), kernel)
            if f > pbest_fit[i]:
                pbest_fit[i], pbest_pos[i] = f, pos[i].copy()
            if f > gbest_fit:
                gbest_fit, gbest_pos = f, pos[i].copy()
        history.append(gbest_fit)
    return gbest_pos, gbest_fit, history


def make_svc(kernel, bp):
    kw = {"kernel": kernel, "C": bp["C"], "random_state": RANDOM_STATE}
    if "gamma" in bp:
        kw["gamma"] = bp["gamma"]
    if "d" in bp:
        kw["degree"] = int(bp["d"])
    if "r" in bp:
        kw["coef0"] = bp["r"]
    return SVC(**kw)


# Jalankan PSO untuk tiap kernel, latih model final, evaluasi
hasil_pso, best_params, best_fitness, histories, y_pred_pso = {}, {}, {}, {}, {}
for kernel in kernels:
    print(f"PSO kernel {kernel} ...")
    best_vec, best_fit, histories[kernel] = run_pso(kernel)
    bp = to_params(best_vec, kernel)
    if "d" in bp:
        bp["d"] = int(round(bp["d"]))
    model = make_svc(kernel, bp).fit(X_train, y_train)
    y_pred_pso[kernel] = model.predict(X_test)
    hasil_pso[kernel] = metrik(y_test, y_pred_pso[kernel])
    best_params[kernel], best_fitness[kernel] = bp, best_fit

# Pilih kernel terbaik
best_kernel = max(kernels, key=lambda k: hasil_pso[k]["Accuracy"])
if hasil_pso["rbf"]["Accuracy"] == hasil_pso[best_kernel]["Accuracy"]:
    best_kernel = "rbf"
bp = best_params[best_kernel]
model_terbaik = make_svc(best_kernel, bp).fit(X_train, y_train)
joblib.dump(model_terbaik, args.model_out)
y_pred_terbaik = model_terbaik.predict(X_test)
y_pred_default_terbaik = y_pred_default[best_kernel]

# Ringkasan hasil
print(f"\nData: {len(df)} sampel | latih {len(X_train)} | uji {len(X_test)}")
print(f"Threshold berat: {batas_01:.1f} g / {batas_12:.1f} g | silhouette {sil_berat:.3f} | kesesuaian {akurasi_berat:.1%}")
print(f"{'Metode':<26}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
rows = [("K-Means", hasil_km)] + [(f"SVM default ({k})", hasil_default[k]) for k in kernels] + [(f"SVM-PSO ({k})", hasil_pso[k]) for k in kernels]
for name, m in rows:
    print(f"{name:<26}{m['Accuracy']:>10.4f}{m['Precision']:>11.4f}{m['Recall']:>9.4f}{m['F1-Score']:>9.4f}")
for k in kernels:
    print(f"PSO {k:<8} fitness {best_fitness[k]:.4f} params {best_params[k]}")
print(f"Kernel terbaik: {best_kernel} -> {args.model_out}")

# Gambar 4.1 Sebaran Data Tomat
colors_cls = {0: "#3b82f6", 1: "#f59e0b", 2: "#ef4444"}
names_cls = {0: "Tidak Matang", 1: "Setengah Matang", 2: "Matang"}
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
for cls in [0, 1, 2]:
    mask = y == cls
    ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=colors_cls[cls], label=names_cls[cls],
               alpha=0.7, s=35, edgecolors="white", linewidths=0.3)
ax.set_xlabel("R_norm", labelpad=14)
ax.set_ylabel("G_norm", labelpad=14)
ax.set_zlabel("B_norm", labelpad=20)
ax.set_title("Sebaran Data Tomat (3D)", pad=20, fontweight="bold")
ax.view_init(elev=25, azim=210)
ax.legend(loc="upper left")
fig.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.93)
plt.savefig("gambar_4_1_scatter3d.png", dpi=600, facecolor="white")
plt.show() if args.show else plt.close()

# Gambar 4.2 Kurva Konvergensi Fitness Tiga Kernel SVM-PSO
fig, ax = plt.subplots(figsize=(9, 5))
ck = {"poly": "#3b82f6", "sigmoid": "#f59e0b", "rbf": "#ef4444"}
for k in kernels:
    ax.plot(range(len(histories[k])), histories[k], label=f"Kernel {k}", color=ck[k], linewidth=2)
ax.set_xlabel("Iterasi")
ax.set_ylabel("Fitness (CV Accuracy)")
ax.set_title("Kurva Konvergensi Fitness Tiga Kernel SVM-PSO", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("gambar_4_2_konvergensi.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show() if args.show else plt.close()

# Gambar 4.3 Perbandingan Performa K-Means, SVM, SVM-PSO
metode = ["K-Means", f"SVM ({best_kernel}, default)", f"SVM + PSO ({best_kernel})"]
metrik_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
vals = np.array([[hasil_km[m] for m in metrik_cols],
                 [hasil_default[best_kernel][m] for m in metrik_cols],
                 [hasil_pso[best_kernel][m] for m in metrik_cols]])
stds = np.array([bootstrap_std(y_test, y_pred_km),
                 bootstrap_std(y_test, y_pred_default_terbaik),
                 bootstrap_std(y_test, y_pred_terbaik)])

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(4)
w = 0.27
bar_colors = ["#94a3b8", "#3b82f6", "#10b981"]
for i, (m, c) in enumerate(zip(metode, bar_colors)):
    ax.bar(x + (i - 1) * w, vals[i], w, yerr=stds[i], capsize=4, label=m, color=c, alpha=0.85,
           error_kw={"elinewidth": 1.2, "ecolor": "#333333"})
    for j, v in enumerate(vals[i]):
        ax.text(x[j] + (i - 1) * w, v + stds[i][j] + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(metrik_cols)
ax.set_ylabel("Skor")
ax.set_title("Perbandingan Performa K-Means, SVM, dan SVM-PSO", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3, axis="y")
ax.set_ylim([0.65, 1.05])
plt.tight_layout()
plt.savefig("gambar_4_3_perbandingan.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show() if args.show else plt.close()

# Gambar 4.4 Confusion Matrix K-Means, SVM Default, SVM-PSO
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
data_cm = [("K-Means", y_pred_km),
           (f"SVM ({best_kernel}, default)", y_pred_default_terbaik),
           (f"SVM + PSO ({best_kernel})", y_pred_terbaik)]
sl = ["TM", "SM", "M"]
for ax, (name, yp) in zip(axes, data_cm):
    cm = confusion_matrix(y_test, yp)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(sl)
    ax.set_yticklabels(sl)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(f"{name}\n(Acc={accuracy_score(y_test, yp):.3f})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
plt.tight_layout()
plt.savefig("gambar_4_4_confusion_matrix.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show() if args.show else plt.close()
