# kaggle.py
# Click-and-run baseline for "AI-Based Modeling for Energy-Efficient Buildings"
# - Auto-extracts 2024_H1_Data.zip, 2024_H2_Data.zip, 2025_H1_Data.zip (if present)
# - Also extracts RBHU-YYYY-MM.zip inside ./extracted
# - Trains on all months <= 2025-05; predicts for 2025-06..07 (10-min grid)
# - Robust to duplicate sensor columns across months (merged by name)
# - Enforces 3h lead (18*10min), causal fill, train-median backfill

from __future__ import annotations
import os, sys, re, zipfile, glob, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# -------- LightGBM (auto-install if missing) --------
try:
    import lightgbm as lgb
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb

# =========================
# Config
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR / "extracted"
ZIP_BUNDLES = ["2024_H1_Data.zip", "2024_H2_Data.zip", "2025_H1_Data.zip"]

BUILDING_ID      = "B205"
TARGET_OBJECT_ID = "B205WC000.AM02"

RESAMPLE_FREQ = "10min"
LEAD_MINUTES  = 180
LEAD_STEPS    = LEAD_MINUTES // 10              # 18
MAX_SENSORS   = 200
MIN_TRAIN_COVERAGE = 0.60

ROLL_WINDOWS = [3, 6, 12, 36]                   # 10-min steps
LAG_STEPS    = [1, 2, 3, 6, 12, 36]
TOP_K_SENSORS = 40

LGB_PARAMS = dict(
    objective="regression",
    metric=["l1", "l2"],
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=50,
    n_estimators=5000,
    verbosity=-1,
    random_state=42,
)

# =========================
# Extraction & months
# =========================
def safe_extract_zip(zippath: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zippath.name} -> {dest} …")
    with zipfile.ZipFile(zippath, "r") as zf:
        zf.extractall(dest)

def auto_extract_bundles() -> None:
    for name in ZIP_BUNDLES:
        zp = SCRIPT_DIR / name
        if zp.exists():
            safe_extract_zip(zp, ROOT_DIR)
    # Month-level zips inside ROOT_DIR
    for zp in sorted(ROOT_DIR.glob("RBHU-20*.zip")):
        month_dir = ROOT_DIR / zp.stem
        if month_dir.exists() and any(month_dir.iterdir()):
            continue
        safe_extract_zip(zp, month_dir)

def find_month_folders(root_dir: Path) -> List[Path]:
    return [p for p in sorted(root_dir.glob("RBHU-20*")) if p.is_dir()]

def parse_year_month_from_name(name: str) -> Tuple[int, int]:
    m = re.search(r"(20\d{2})[-_](0[1-9]|1[0-2])", name)
    if not m:
        raise ValueError(f"Cannot parse year-month from: {name}")
    return int(m.group(1)), int(m.group(2))

# =========================
# IO & panel building
# =========================
def list_parquet_files_for_building(month_dir: Path, building_id: str) -> List[Path]:
    pats = [
        month_dir / "RBHU" / building_id / "**" / "*.parquet",
        month_dir / "**" / building_id / "**" / "*.parquet",
    ]
    files: List[Path] = []
    for pat in pats:
        files.extend([Path(p) for p in glob.glob(str(pat), recursive=True)])
    # de-dup paths
    seen = set(); out = []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    return out

def read_metadata_if_any(root_dir: Path) -> pd.DataFrame:
    for name in ["metadata.parquet", "metadata.xlsx"]:
        p = root_dir / name
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    return pd.read_parquet(p)
                else:
                    return pd.read_excel(p)
            except Exception:
                pass
    print("[WARN] metadata not found — proceeding without it.")
    return pd.DataFrame()

def load_and_resample_sensor(parquet_path: Path, freq: str) -> pd.Series:
    df = pd.read_parquet(parquet_path)
    # timestamp column
    ts_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "time" in lc or "timestamp" in lc:
            ts_col = c; break
    if ts_col is None:
        ts_col = df.columns[0]
    val_cols = [c for c in df.columns if c != ts_col]
    if not val_cols:
        raise ValueError(f"No value column in {parquet_path}")
    val = val_cols[0]

    s = df[[ts_col, val]].copy()
    s[ts_col] = pd.to_datetime(s[ts_col], utc=True, errors="coerce")
    s = s.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    # make tz-naive UTC
    try:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    except Exception:
        s.index = s.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT").tz_localize(None)
    s = s.resample(freq).ffill()
    return s[val].rename(parquet_path.stem)

def build_panel(all_files: List[Path], freq: str) -> pd.DataFrame:
    """
    Load many sensor files and MERGE series with the same name into one column.
    This avoids duplicate column-name traps that lead to 'truth value of a Series is ambiguous'.
    """
    merged: Dict[str, pd.Series] = {}
    for fp in all_files:
        try:
            ser = load_and_resample_sensor(fp, freq)
        except Exception:
            continue
        col = ser.name  # e.g., 'B205WC000.AM02'
        if col in merged:
            # Union the index and combine causally (prefer existing non-NaNs)
            a = merged[col]
            # Fast path: if indices don't overlap monthly, combine_first is safe
            merged[col] = a.combine_first(ser)
        else:
            merged[col] = ser

    if not merged:
        raise RuntimeError("No usable sensors loaded. Check paths.")

    df = pd.DataFrame(merged).sort_index()
    # Drop columns entirely NA (handle as Series -> bool safely)
    drop_cols = [c for c in df.columns if bool(df[c].isna().all())]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df

def reindex_to_full_grid(panel: pd.DataFrame, months: List[Path]) -> pd.DataFrame:
    ym = sorted([parse_year_month_from_name(m.name) for m in months])
    y0, m0 = ym[0]; y1, m1 = ym[-1]
    start = pd.Timestamp(f"{y0:04d}-{m0:02d}-01 00:00:00")
    end   = (pd.Timestamp(f"{y1:04d}-{m1:02d}-01") + pd.offsets.MonthEnd(1)).replace(hour=23, minute=50, second=0)
    idx = pd.date_range(start, end, freq=RESAMPLE_FREQ)
    out = panel.reindex(idx)
    print(f"[INFO] Reindexed to full grid: {out.shape} (from {start} to {end})")
    return out

# =========================
# Features & selection
# =========================
def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    tf = pd.DataFrame(index=index)
    tf["minute"] = index.minute
    tf["hour"]   = index.hour
    tf["day"]    = index.day
    tf["dow"]    = index.dayofweek
    tf["week"]   = index.isocalendar().week.astype(int)
    tf["month"]  = index.month
    tf["is_weekend"] = (tf["dow"] >= 5).astype(int)
    return tf

def add_lag_roll_feats(df: pd.DataFrame,
                       base_cols: List[str],
                       roll_windows: List[int],
                       lag_steps: List[int]) -> pd.DataFrame:
    X = df.copy()
    for c in base_cols:
        for lag in lag_steps:
            X[f"{c}_lag{lag}"] = X[c].shift(lag)
        for w in roll_windows:
            X[f"{c}_rollmean{w}"] = X[c].rolling(w).mean()
            X[f"{c}_rollstd{w}"]  = X[c].rolling(w).std()
    return X

def select_predictor_columns_train_pairwise(panel_train: pd.DataFrame,
                                            target_col: str,
                                            k: int = 40,
                                            min_pairs: int = 500) -> List[str]:
    cors: Dict[str, float] = {}
    tgt = panel_train[target_col]
    for c in panel_train.columns:
        if c == target_col:
            continue
        s = pd.concat([panel_train[c], tgt], axis=1).dropna()
        if len(s) < min_pairs:
            continue
        cors[c] = s[c].corr(s[target_col])
    if not cors:
        var = panel_train.drop(columns=[target_col], errors="ignore").var(numeric_only=True).sort_values(ascending=False)
        return list(var.index[:min(k, len(var))])
    top = sorted(cors.items(), key=lambda x: -abs(x[1]))[:k]
    return [c for c, _ in top]

# =========================
# Submission grid
# =========================
def build_submission_grid() -> pd.DatetimeIndex:
    start = pd.Timestamp("2025-06-01 00:00:00")
    end   = pd.Timestamp("2025-07-31 23:50:00")
    return pd.date_range(start, end, freq=RESAMPLE_FREQ)

def make_submission(model,
                    features: List[str],
                    X_all_shifted: pd.DataFrame,
                    train_medians: pd.Series,
                    out_path: Path) -> pd.DataFrame:
    grid = build_submission_grid()
    Xg = X_all_shifted.reindex(grid)
    Xg = Xg.ffill().fillna(train_medians)
    preds = model.predict(Xg[features])
    sub = pd.DataFrame({
        "ID": [ts.strftime("%Y-%m-%d_%H:%M:%S") for ts in grid],
        "TARGET_VARIABLE": preds,
    })
    sub.to_csv(out_path, index=False)
    print(f"[OK] Saved submission to {out_path} with {len(sub)} rows (expected 8784).")
    return sub

# =========================
# Main
# =========================
def main():
    auto_extract_bundles()

    months = find_month_folders(ROOT_DIR)
    if not months:
        raise FileNotFoundError(f"No RBHU-YYYY-MM folders under {ROOT_DIR}")
    months_sorted = sorted(months, key=lambda p: parse_year_month_from_name(p.name))
    print(f"[INFO] Months found: {[m.name for m in months_sorted]}")

    train_months = [m for m in months_sorted if parse_year_month_from_name(m.name) <= (2025, 5)]
    print(f"[INFO] Train months: {[m.name for m in train_months]}")

    print(f"[INFO] Root: {ROOT_DIR}")
    print(f"[INFO] Building: {BUILDING_ID} | Target: {TARGET_OBJECT_ID}")
    print(f"[INFO] Resample: {RESAMPLE_FREQ} | Lead: {LEAD_MINUTES} min | Max sensors: {MAX_SENSORS} | Min cov (train): {MIN_TRAIN_COVERAGE}")

    meta = read_metadata_if_any(ROOT_DIR)

    # Gather files for building across all months
    all_files: List[Path] = []
    for m in months_sorted:
        files = list_parquet_files_for_building(m, BUILDING_ID)
        print(f"[INFO] {m.name}: found {len(files)} parquet files for {BUILDING_ID}")
        all_files.extend(files)
    if not all_files:
        raise RuntimeError("No parquet files found for building.")

    # Load + resample and merge duplicates by sensor name
    panel_raw = build_panel(all_files, RESAMPLE_FREQ)

    # Reindex to continuous grid across all months
    panel = reindex_to_full_grid(panel_raw, months_sorted)

    # Ensure target present
    if TARGET_OBJECT_ID not in panel.columns:
        candidates = [c for c in panel.columns if c.lower() == TARGET_OBJECT_ID.lower()]
        if candidates:
            target_col = candidates[0]
        else:
            raise RuntimeError(f"Target column {TARGET_OBJECT_ID} not found in panel.")
    else:
        target_col = TARGET_OBJECT_ID
    print(f"[INFO] Target column: {target_col}")

    # Drop columns exactly equal to target
    dup_cols = [c for c in panel.columns if c != target_col and panel[c].equals(panel[target_col])]
    if dup_cols:
        panel = panel.drop(columns=dup_cols, errors="ignore")
        print(f"[INFO] Dropped {len(dup_cols)} exact-duplicate columns of target.")

    # Train mask
    train_mask = pd.Series(False, index=panel.index)
    for (y, m) in [parse_year_month_from_name(tm.name) for tm in train_months]:
        train_mask |= ((panel.index.year == y) & (panel.index.month == m))

    # Coverage filtering on TRAIN only
    train_panel = panel.loc[train_mask]
    coverages = train_panel.notna().mean(axis=0)
    keep_cols = [c for c in panel.columns if c == target_col or coverages.get(c, 0.0) >= MIN_TRAIN_COVERAGE]
    if len(keep_cols) <= 1:
        print("[WARN] No columns met train coverage; keeping all.")
        kept = panel.columns.tolist()
    else:
        kept = keep_cols
    if len(kept) > MAX_SENSORS:
        others = [c for c in kept if c != target_col]
        by_cov = sorted(others, key=lambda c: coverages.get(c, 0.0), reverse=True)[:MAX_SENSORS-1]
        kept = [target_col] + by_cov
    panel = panel[kept]
    print(f"[INFO] Panel shape after TRAIN coverage & cap: {panel.shape}")

    # Time features
    tf = time_features(panel.index)

    # Feature selection on TRAIN only (pairwise corr)
    top_sensors = select_predictor_columns_train_pairwise(
        train_panel[panel.columns], target_col,
        k=min(TOP_K_SENSORS, max(1, panel.shape[1]-1)),
        min_pairs=300
    )
    if not top_sensors:
        print("[WARN] Corr-selection empty; using variance top 10.")
        var = train_panel.drop(columns=[target_col], errors="ignore").var(numeric_only=True).sort_values(ascending=False)
        top_sensors = list(var.index[:10])
    print(f"[INFO] Using {len(top_sensors)} sensors for lags/rolls.")

    # Build features
    base_for_lags = list(top_sensors)
    X_base = pd.concat([panel[base_for_lags], tf], axis=1) if base_for_lags else tf.copy()
    X_all = add_lag_roll_feats(X_base, base_for_lags, ROLL_WINDOWS, LAG_STEPS)
    X_all_shifted = X_all.shift(LEAD_STEPS)  # 3h lead

    # Train/valid frames
    X_train_full = X_all_shifted.loc[train_mask].copy()
    y_train_full = panel.loc[train_mask, target_col].copy()
    keep = y_train_full.notna()
    X_train_full = X_train_full[keep]
    y_train_full = y_train_full[keep]

    if len(X_train_full) == 0:
        print("[WARN] Training frame empty after lags/rolls; falling back to time-features-only.")
        X_train_full = tf.loc[train_mask].copy()
        X_train_full = X_train_full[keep]
        if len(X_train_full) == 0:
            raise RuntimeError("No training rows available (target missing in train months).")

    X_train_full = X_train_full.sort_index()
    y_train_full = y_train_full.sort_index()
    n = len(X_train_full)
    cut = max(1, int(n * 0.8))
    X_tr = X_train_full.iloc[:cut].copy()
    y_tr = y_train_full.iloc[:cut].copy()
    X_va = X_train_full.iloc[cut:].copy()
    y_va = y_train_full.iloc[cut:].copy()

    # Causal fill then train medians
    X_tr = X_tr.ffill()
    train_medians = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(train_medians)
    X_va = X_va.ffill().fillna(train_medians)

    if X_tr.empty or y_tr.empty:
        raise RuntimeError("Training set empty after NA handling.")

    print(f"[INFO] Train: {X_tr.shape} | Valid: {X_va.shape}")

    # Train model
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    if len(X_va) > 0:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric=["l1", "l2"],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(50)]
        )
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        va_pred = model.predict(X_va)
        mae  = mean_absolute_error(y_va, va_pred)
        rmse = mean_squared_error(y_va, va_pred, squared=False)
        print(f"[VALID] MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    else:
        model.fit(X_tr, y_tr)
        print("[INFO] Trained on all training data (no validation split).")

    # Submission
    feature_cols = X_tr.columns.tolist()
    out_csv = SCRIPT_DIR / "submission.csv"
    _ = make_submission(model, feature_cols, X_all_shifted, train_medians, out_csv)

    # Optional: feature importance
    try:
        import matplotlib.pyplot as plt
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)[:30]
        plt.figure(figsize=(8, 10))
        imp.iloc[::-1].plot(kind="barh")
        plt.title("Top 30 Feature Importances (LightGBM)")
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / "feature_importance.png", dpi=160)
        print("[OK] Saved feature_importance.png")
    except Exception as e:
        print(f"[WARN] Could not save feature importance: {e}")

if __name__ == "__main__":
    main()
