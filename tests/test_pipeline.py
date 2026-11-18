import numpy as np
import pandas as pd

from tomato.features import MIN_CLEAR, normalize_rgbc, preprocess
from tomato.models import KMeansClassifier, decode
from tomato.pso import pso_maximize
from tomato.train import main


def test_normalize_divides_by_clear():
    r, g, b = normalize_rgbc(500, 250, 100, 1000)
    assert (r, g, b) == (0.5, 0.25, 0.1)


def test_preprocess_drops_low_clear():
    df = pd.DataFrame({"r": [1, 1], "g": [1, 1], "b": [1, 1], "c": [MIN_CLEAR - 1, MIN_CLEAR], "label": ["ripe", "ripe"]})
    assert len(preprocess(df)) == 1


def test_pso_finds_maximum_of_simple_function():
    res = pso_maximize(lambda p: -((p[0] - 3) ** 2 + (p[1] + 1) ** 2), np.array([-10, -10]), np.array([10, 10]), n_iter=60)
    assert np.allclose(res.best_position, [3, -1], atol=0.05)
    assert res.history == sorted(res.history)  # global best never gets worse


def test_decode_rounds_degree():
    assert decode("poly", np.array([1.0, 0.1, 0.0, 2.6]))["degree"] == 3


def test_kmeans_maps_clusters_to_labels():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 0.01, (20, 3)) for m in (0.2, 0.5, 0.8)])
    y = np.repeat([0, 1, 2], 20)
    assert (KMeansClassifier().fit(X, y).predict(X) == y).mean() > 0.95


def test_end_to_end_on_sample_data(tmp_path):
    main(["--data", "data/sample_readings.csv", "--particles", "4", "--iters", "3", "--bootstrap", "20",
          "--kernels", "rbf", "--results", str(tmp_path / "res"), "--model-out", str(tmp_path / "m.joblib")])
    assert (tmp_path / "res" / "metrics.json").exists()
    assert (tmp_path / "m.joblib").exists()


def test_thesis_sorter_runs_in_simulation(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    main(["--data", str(root / "data/sample_readings.csv"), "--particles", "3", "--iters", "2", "--bootstrap", "10",
          "--kernels", "rbf", "--results", str(tmp_path / "res"), "--model-out", str(tmp_path / "m.joblib")])
    out = subprocess.run([sys.executable, str(root / "deploy/sorter.py"), "--model", str(tmp_path / "m.joblib"),
                          "--simulate", "--items", "3"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "Total: 3 tomat" in out.stdout
    assert list(tmp_path.glob("log_pengujian_*.csv"))
