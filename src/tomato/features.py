"""Preprocessing exactly as in the thesis (Section 3.6 / 4.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tomato import CLASSES, FEATURES

# Readings with a Clear channel below this mean too little light or no object.
MIN_CLEAR = 500

# Raw labels used during data collection -> canonical English labels.
LABEL_ALIASES = {
    "tidak matang": "unripe",
    "tidak_matang": "unripe",
    "mentah": "unripe",
    "unripe": "unripe",
    "setengah matang": "partially_ripe",
    "setengah_matang": "partially_ripe",
    "partially ripe": "partially_ripe",
    "partially_ripe": "partially_ripe",
    "matang": "ripe",
    "ripe": "ripe",
}


def normalize_rgbc(r, g, b, c):
    """Divide R, G, B by the Clear channel to remove light-intensity effects."""
    c = np.asarray(c, dtype=float)
    return np.asarray(r) / c, np.asarray(g) / c, np.asarray(b) / c


def load_readings(path: str, sheet: str = "Data Gabungan") -> pd.DataFrame:
    """Load readings from CSV or Excel.

    Accepts raw channels (R, G, B, C) and/or already-normalized ones (R_norm, G_norm, B_norm),
    plus a label column (Label / label) and optionally fruit_id. The thesis file is
    dataset_tomat.xlsx, sheet 'Data Gabungan'."""
    df = pd.read_excel(path, sheet_name=sheet) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    has_raw = {"r", "g", "b", "c"} <= set(df.columns)
    has_norm = set(FEATURES) <= set(df.columns)
    if "label" not in df.columns or not (has_raw or has_norm):
        raise ValueError(f"{path} needs a label column plus R,G,B,C or R_norm,G_norm,B_norm")
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    if df["label"].isna().any():
        raise ValueError("Unknown labels found; extend LABEL_ALIASES in features.py")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Filter invalid readings and add normalized features."""
    if "c" in df.columns:
        out = df[df["c"] >= MIN_CLEAR].copy()
        out["r_norm"], out["g_norm"], out["b_norm"] = normalize_rgbc(out["r"], out["g"], out["b"], out["c"])
    else:  # already normalized (thesis spreadsheet); invalid readings were filtered at collection time
        out = df.copy()
    out["y"] = out["label"].map({name: i for i, name in enumerate(CLASSES)})
    return out.reset_index(drop=True)


def xy(df: pd.DataFrame):
    return df[FEATURES].to_numpy(), df["y"].to_numpy()
