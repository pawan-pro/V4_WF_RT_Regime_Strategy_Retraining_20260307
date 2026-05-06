# Databricks notebook source
"""MUB-27 V4.2 strict-PIT label redesign.

Research-only label redesign pass. Outputs are isolated under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_label_redesign_strict_pit_2026-05-06/`

No production V4 Delta tables, Saxo execution, token refresh, RT/dashboard
notebooks, live schedules, model registry entries, or trading logic are
modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
OUTPUT_ROOT = BASE / "v4_2_label_redesign_strict_pit_2026-05-06"
LABELS_DIR = OUTPUT_ROOT / "labels"
SIGNALS_DIR = OUTPUT_ROOT / "signals"
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"

FAITHFUL_GRID_PATH = (
    BASE / "v4_1_csv" / "full_legacy_grid_clone_2026-04-30" / "grids" / "full_legacy_grid_clone.csv"
)
DEFENSIVE_PRICE_ROOT = BASE / "v4_2_defensive_sleeve_prices_2026-05-06"
DEFENSIVE_PRICE_PATH = DEFENSIVE_PRICE_ROOT / "prices" / "defensive_etf_hourly_prices.csv"
STRICT_ROOT = BASE / "v4_2_strict_pit_research_2026-05-06"
BILSTM_ROOT = BASE / "v4_2_bilstm_strict_pit_2026-05-06"
TFT_ROOT = BASE / "v4_2_tft_strict_pit_2026-05-06"
WINDOW_ROOT = BASE / "v4_2_tft_window_defensive_sleeve_2026-05-06"
LEVERAGE_ROOT = BASE / "v4_2_leverage_sensitivity_2026-05-06"

B_C2_SIGNAL_PATH = STRICT_ROOT / "signals" / "B_reference_teacher_smoothed_signals.csv"
RAW_WF_SIGNAL_PATH = STRICT_ROOT / "signals" / "A_raw_label_walkforward_signals.csv"
WINDOW_COMPARISON_PATH = WINDOW_ROOT / "forensics" / "full_variant_comparison.csv"

TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FINAL_FOLD_END = pd.Timestamp("2026-01-01")
FWD_HOURS = 168
INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005
RANDOM_SEED = 42
SEQUENCE_WINDOW = 72
MAX_TRAIN_SAMPLES_PER_CLASS = 2500
MAX_EPOCHS = 3
PATIENCE = 1
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SPY_CAGR = 0.127546
B_C2_CAGR = 0.208405
REGIME_NAME = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "BIL"}

FEATURES = [
    "CPIAUCSL_z",
    "PCEPI_z",
    "PCE_z",
    "PAYEMS_z",
    "UNRATE_z",
    "GDPC1_z",
    "INDPRO_z",
    "UMCSENT_z",
    "VIXCLS_z",
    "vix3m_z",
    "VIX_Inversion_Z",
    "Price_Mom_24H",
    "CPI_Event_Active",
    "NFP_Event_Active",
]


def ensure_python_deps():
    missing = []
    for module_name, package_name in [("xgboost", "xgboost"), ("torch", "torch"), ("sklearn", "scikit-learn")]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    import sklearn
    import torch
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    return torch, sklearn, xgb, StandardScaler


torch, sklearn, xgb, StandardScaler = ensure_python_deps()
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class LabelSpec:
    name: str
    family: str
    params: dict[str, Any]


LABEL_SPECS = [
    LabelSpec("raw_forward_return_baseline", "raw_forward_return_baseline", {"upper": 0.005, "lower": -0.005}),
    LabelSpec("vol_adj_thr_025", "volatility_adjusted_forward_label", {"threshold": 0.25}),
    LabelSpec("vol_adj_thr_050", "volatility_adjusted_forward_label", {"threshold": 0.50}),
    LabelSpec("vol_adj_thr_075", "volatility_adjusted_forward_label", {"threshold": 0.75}),
    LabelSpec("drawdown_aware_dd_010_ret_005", "drawdown_aware_label", {"dd_threshold": -0.010, "strong_neg": -0.005}),
    LabelSpec("drawdown_aware_dd_020_ret_010", "drawdown_aware_label", {"dd_threshold": -0.020, "strong_neg": -0.010}),
    LabelSpec("triple_barrier_005", "triple_barrier_label", {"upper": 0.005, "lower": -0.005}),
    LabelSpec("triple_barrier_010", "triple_barrier_label", {"upper": 0.010, "lower": -0.010}),
    LabelSpec("triple_barrier_vol_050", "triple_barrier_label", {"vol_mult": 0.50}),
]


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dirs() -> None:
    for path in [LABELS_DIR, SIGNALS_DIR, PERFORMANCE_DIR, FORENSICS_DIR, MANIFESTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_output_path(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside isolated research output root: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_text(text: str, path: Path) -> None:
    assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(payload: dict[str, Any], path: Path) -> None:
    assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.sort_index(axis=1).to_csv(index=False).encode("utf-8")).hexdigest()


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    try:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    except Exception:
        pass
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return normalize_timestamp(pd.read_csv(path, **kwargs))


def monthly_folds() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds = []
    start = EVAL_START
    while start < FINAL_FOLD_END:
        end = start + relativedelta(months=1)
        folds.append((start, end))
        start = end
    return folds


def label_from_return(ret: float, upper: float = 0.005, lower: float = -0.005) -> float:
    if pd.isna(ret):
        return np.nan
    if ret > upper:
        return 0
    if ret < lower:
        return 2
    return 1


def first_barrier_label(path: np.ndarray, upper: float, lower: float) -> float:
    if len(path) == 0 or np.isnan(path).all():
        return np.nan
    up = np.flatnonzero(path >= upper)
    dn = np.flatnonzero(path <= lower)
    first_up = up[0] if len(up) else None
    first_dn = dn[0] if len(dn) else None
    if first_up is None and first_dn is None:
        return 1
    if first_dn is None or (first_up is not None and first_up < first_dn):
        return 0
    return 2


def build_labels(grid: pd.DataFrame) -> pd.DataFrame:
    close = grid["SPY_close"].to_numpy(dtype=float)
    hourly_ret = pd.Series(close).pct_change().to_numpy(dtype=float)
    trailing_vol = pd.Series(hourly_ret).rolling(FWD_HOURS, min_periods=48).std().to_numpy() * math.sqrt(FWD_HOURS)
    fwd_ret = np.full(len(grid), np.nan)
    fwd_max_dd = np.full(len(grid), np.nan)
    tb_005 = np.full(len(grid), np.nan)
    tb_010 = np.full(len(grid), np.nan)
    tb_vol = np.full(len(grid), np.nan)
    for i in range(0, max(0, len(grid) - FWD_HOURS)):
        start = close[i]
        future = close[i + 1 : i + FWD_HOURS + 1]
        if not np.isfinite(start) or start <= 0 or len(future) < FWD_HOURS:
            continue
        path = future / start - 1.0
        fwd_ret[i] = close[i + FWD_HOURS] / start - 1.0
        running_peak = np.maximum.accumulate(np.r_[start, future])
        dd_path = np.r_[start, future] / running_peak - 1.0
        fwd_max_dd[i] = float(np.min(dd_path[1:]))
        tb_005[i] = first_barrier_label(path, 0.005, -0.005)
        tb_010[i] = first_barrier_label(path, 0.010, -0.010)
        vol_barrier = trailing_vol[i] * 0.50 if np.isfinite(trailing_vol[i]) else np.nan
        if np.isfinite(vol_barrier) and vol_barrier > 0:
            tb_vol[i] = first_barrier_label(path, vol_barrier, -vol_barrier)

    labels = pd.DataFrame(
        {
            "timestamp": grid["timestamp"],
            "fwd_return_1w": fwd_ret,
            "trailing_vol_1w": trailing_vol,
            "vol_adj_score": fwd_ret / trailing_vol,
            "future_max_drawdown_1w": fwd_max_dd,
            "label_available_ts": grid["timestamp"].shift(-FWD_HOURS),
        }
    )
    labels["raw_forward_return_baseline"] = labels["fwd_return_1w"].apply(label_from_return)
    for threshold in [0.25, 0.50, 0.75]:
        name = f"vol_adj_thr_{int(threshold * 100):03d}"
        score = labels["vol_adj_score"]
        labels[name] = np.select([score > threshold, score < -threshold], [0, 2], default=1).astype(float)
        labels.loc[score.isna(), name] = np.nan
    for dd_thr, strong_neg, name in [
        (-0.010, -0.005, "drawdown_aware_dd_010_ret_005"),
        (-0.020, -0.010, "drawdown_aware_dd_020_ret_010"),
    ]:
        labels[name] = np.select(
            [
                (labels["fwd_return_1w"] > 0) & (labels["future_max_drawdown_1w"] > dd_thr),
                (labels["future_max_drawdown_1w"] <= dd_thr) | (labels["fwd_return_1w"] < strong_neg),
            ],
            [0, 2],
            default=1,
        ).astype(float)
        labels.loc[labels["fwd_return_1w"].isna() | labels["future_max_drawdown_1w"].isna(), name] = np.nan
    labels["triple_barrier_005"] = tb_005
    labels["triple_barrier_010"] = tb_010
    labels["triple_barrier_vol_050"] = tb_vol
    return labels


def load_grid_and_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = read_csv(FAITHFUL_GRID_PATH)
    cols = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]
    missing = [c for c in cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Faithful grid missing required columns: {missing}")
    grid = grid[cols].copy()
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid["feature_ok"] = grid[FEATURES + ["SPY_close", "QQQ_close"]].notna().all(axis=1)
    grid["row_pos"] = np.arange(len(grid))
    labels = build_labels(grid)
    out = grid.merge(labels, on="timestamp", how="left")
    return out, labels


def working_frame(grid: pd.DataFrame, label_col: str) -> pd.DataFrame:
    need = ["timestamp", "SPY_close", "QQQ_close", "feature_ok", "label_available_ts", label_col, *FEATURES]
    out = grid[need].copy()
    out = out[(out["timestamp"] >= TRAIN_START) & out["feature_ok"] & out[label_col].notna()].copy()
    out[label_col] = out[label_col].astype(int)
    out = out.reset_index(drop=True)
    out["work_pos"] = np.arange(len(out))
    return out


def load_bil_prices() -> pd.DataFrame:
    prices = read_csv(DEFENSIVE_PRICE_PATH)
    prices = prices[prices["symbol"].astype(str).str.upper().eq("BIL")].copy()
    if prices.empty:
        raise ValueError("No BIL prices in validated defensive ETF price file")
    out = prices[["timestamp", "close"]].rename(columns={"close": "defensive_close"}).copy()
    out["defensive_close"] = pd.to_numeric(out["defensive_close"], errors="coerce")
    return out.dropna(subset=["defensive_close"]).sort_values("timestamp").reset_index(drop=True)


def load_reference_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    b = read_csv(B_C2_SIGNAL_PATH)
    raw = read_csv(RAW_WF_SIGNAL_PATH)
    for df in [b, raw]:
        df["predicted_regime"] = pd.to_numeric(df["predicted_regime"], errors="coerce").astype("Int64")
        df.dropna(subset=["predicted_regime"], inplace=True)
        df["predicted_regime"] = df["predicted_regime"].astype(int)
    return b, raw


def canonical_eval(work: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    ref_ts = set(ref["timestamp"])
    out = work[(work["timestamp"] >= EVAL_START) & (work["timestamp"] <= EVAL_END) & work["timestamp"].isin(ref_ts)].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=3).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return weights[y.astype(int)]


def run_xgboost_label(grid: pd.DataFrame, label_col: str, ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = working_frame(grid, label_col)
    canonical = canonical_eval(work, ref)
    all_features = work[FEATURES].to_numpy(dtype=np.float32)
    labels = work[label_col].to_numpy(dtype=int)
    timestamps = work["timestamp"].reset_index(drop=True)
    label_available = pd.to_datetime(work["label_available_ts"]).reset_index(drop=True)
    rows, manifests = [], []
    for fold_num, (fold_start, fold_end) in enumerate(monthly_folds(), start=1):
        train_mask = ((timestamps < fold_start) & (label_available < fold_start)).to_numpy()
        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        if train_mask.sum() < 1000 or test_df.empty:
            continue
        test_pos = work.index[work["timestamp"].isin(test_df["timestamp"])].to_numpy()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(all_features[train_mask]).astype(np.float32)
        y_train = labels[train_mask]
        x_test = scaler.transform(all_features[test_pos]).astype(np.float32)
        model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            n_estimators=120,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=RANDOM_SEED + fold_num,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=2,
        )
        model.fit(x_train, y_train, sample_weight=balanced_sample_weight(y_train), verbose=False)
        test_df["predicted_regime"] = model.predict(x_test).astype(int)
        sig = test_df[["timestamp", label_col, "predicted_regime"]].copy()
        sig = sig.rename(columns={label_col: "label_regime"})
        sig["variant"] = f"xgboost_{label_col}"
        sig["model_family"] = "xgboost"
        sig["label_family"] = label_col
        sig["regime_name"] = sig["predicted_regime"].map(REGIME_NAME)
        sig["allocation_scale"] = 1.0
        rows.append(sig)
        manifests.append(
            {
                "variant": f"xgboost_{label_col}",
                "fold_start": fold_start,
                "fold_end": fold_end,
                "train_rows": int(train_mask.sum()),
                "test_rows": len(test_df),
                "train_label_distribution": dist(y_train),
                "prediction_distribution": dist(test_df["predicted_regime"]),
                "strict_pit": True,
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(manifests)


def dist(series: pd.Series | np.ndarray) -> dict[str, int]:
    if series is None or len(series) == 0:
        return {}
    vc = pd.Series(series).dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


class TemporalAttentionClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x):
        projected = torch.tanh(self.input_projection(x))
        encoded, _ = self.encoder(projected)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1).unsqueeze(-1)
        return self.head(self.dropout(torch.sum(encoded * weights, dim=1)))


def sequence_tensor(features: np.ndarray, end_positions: np.ndarray, window: int) -> np.ndarray:
    x = np.empty((len(end_positions), window, features.shape[1]), dtype=np.float32)
    for i, pos in enumerate(end_positions):
        x[i] = features[pos - window + 1 : pos + 1]
    return x


def balanced_sample_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for cls in [0, 1, 2]:
        idx = np.flatnonzero(labels == cls)
        if len(idx) > MAX_TRAIN_SAMPLES_PER_CLASS:
            idx = rng.choice(idx, size=MAX_TRAIN_SAMPLES_PER_CLASS, replace=False)
        selected.append(idx)
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=3).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_attention(x_train, y_train, x_val, y_val, device, seed: int) -> TemporalAttentionClassifier:
    seed_everything(seed)
    model = TemporalAttentionClassifier(input_size=x_train.shape[-1], hidden_size=32, dropout=0.2).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val.astype(np.int64))), batch_size=BATCH_SIZE, shuffle=False)
    best_state = None
    best_val = float("inf")
    bad = 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                losses.append(float(loss_fn(model(xb.to(device)), yb.to(device)).detach().cpu()))
        val = float(np.mean(losses)) if losses else float("inf")
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_attention(model: TemporalAttentionClassifier, x_test: np.ndarray, device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(torch.argmax(model(xb.to(device)), dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(int)


def run_tft_label(grid: pd.DataFrame, label_col: str, ref: pd.DataFrame, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = working_frame(grid, label_col)
    canonical = canonical_eval(work, ref)
    all_features = work[FEATURES].to_numpy(dtype=np.float32)
    labels = work[label_col].to_numpy(dtype=int)
    timestamps = work["timestamp"].reset_index(drop=True)
    label_available = pd.to_datetime(work["label_available_ts"]).reset_index(drop=True)
    canonical_pos = dict(zip(work["timestamp"], work["work_pos"]))
    rng = np.random.default_rng(RANDOM_SEED)
    rows, manifests = [], []
    for fold_num, (fold_start, fold_end) in enumerate(monthly_folds(), start=1):
        fit_mask = (timestamps < fold_start).to_numpy()
        train_end_mask = ((timestamps < fold_start) & (label_available < fold_start) & (work["work_pos"] >= SEQUENCE_WINDOW - 1)).to_numpy()
        train_end_positions = np.flatnonzero(train_end_mask)
        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        test_positions = np.array([canonical_pos[t] for t in test_df["timestamp"] if t in canonical_pos], dtype=int)
        test_positions = test_positions[test_positions >= SEQUENCE_WINDOW - 1]
        test_df = test_df[test_df["timestamp"].isin(timestamps.iloc[test_positions].to_list())].copy()
        if len(train_end_positions) < 1200 or len(test_df) == 0:
            continue
        split_idx = int(len(train_end_positions) * 0.85)
        train_pos = train_end_positions[:split_idx]
        val_pos = train_end_positions[split_idx:]
        sample_idx = balanced_sample_indices(labels[train_pos], rng)
        train_pos_sample = train_pos[sample_idx]
        scaler = StandardScaler()
        scaled = scaler.fit(all_features[fit_mask]).transform(all_features).astype(np.float32)
        model = train_attention(
            sequence_tensor(scaled, train_pos_sample, SEQUENCE_WINDOW),
            labels[train_pos_sample],
            sequence_tensor(scaled, val_pos, SEQUENCE_WINDOW),
            labels[val_pos],
            device,
            RANDOM_SEED + fold_num,
        )
        test_df["predicted_regime"] = predict_attention(model, sequence_tensor(scaled, test_positions, SEQUENCE_WINDOW), device)
        sig = test_df[["timestamp", label_col, "predicted_regime"]].copy().rename(columns={label_col: "label_regime"})
        sig["variant"] = f"tft72_bil_{label_col}"
        sig["model_family"] = "tft_attention_72h"
        sig["label_family"] = label_col
        sig["sequence_window_hours"] = SEQUENCE_WINDOW
        sig["regime_name"] = sig["predicted_regime"].map(REGIME_NAME)
        sig["allocation_scale"] = 1.0
        rows.append(sig)
        manifests.append(
            {
                "variant": f"tft72_bil_{label_col}",
                "fold_start": fold_start,
                "fold_end": fold_end,
                "train_sequence_count_before_sampling": len(train_pos),
                "train_sequence_count_after_sampling": len(train_pos_sample),
                "validation_sequence_count": len(val_pos),
                "test_sequence_count": len(test_df),
                "train_label_distribution": dist(labels[train_pos]),
                "prediction_distribution": dist(test_df["predicted_regime"]),
                "strict_pit": True,
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True), pd.DataFrame(manifests)


def price_update_frame(grid: pd.DataFrame, signals: pd.DataFrame, bil_prices: pd.DataFrame) -> pd.DataFrame:
    sig = signals[["timestamp", "predicted_regime", "allocation_scale"]].drop_duplicates("timestamp").copy()
    data = grid[["timestamp", "SPY_close", "QQQ_close"]].merge(sig, on="timestamp", how="inner").sort_values("timestamp")
    data = data.merge(bil_prices[["timestamp", "defensive_close"]], on="timestamp", how="left")
    data = data[(data["timestamp"] >= EVAL_START) & (data["timestamp"] <= EVAL_END)].copy()
    data[["SPY_close", "QQQ_close", "defensive_close"]] = data[["SPY_close", "QQQ_close", "defensive_close"]].ffill()
    data["price_update"] = data[["SPY_close", "QQQ_close"]].diff().abs().sum(axis=1).gt(0)
    if not data.empty:
        data.loc[data.index[0], "price_update"] = True
    data = data[data["price_update"]].dropna(subset=["SPY_close", "QQQ_close", "predicted_regime"]).reset_index(drop=True)
    data["predicted_regime"] = data["predicted_regime"].astype(int)
    data["allocation_scale"] = pd.to_numeric(data["allocation_scale"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    return data


def metrics_from_daily(daily: pd.DataFrame, variant: str) -> dict[str, Any]:
    ret = daily["strategy_equity"].astype(float).pct_change().dropna()
    downside = ret[ret < 0]
    final_equity = float(daily["strategy_equity"].iloc[-1])
    dd = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1.0
    return {
        "variant": variant,
        "CAGR": (final_equity / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
        "Sharpe": ret.mean() / ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0,
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0) if len(downside) > 1 and downside.std(ddof=1) else 0.0,
        "Max DD": float(dd.min()),
        "Final Equity": final_equity,
    }


def run_backtest(grid: pd.DataFrame, signals: pd.DataFrame, variant: str, bil_prices: pd.DataFrame):
    data = price_update_frame(grid, signals, bil_prices)
    equity = [INITIAL_CAPITAL]
    spy_equity = [INITIAL_CAPITAL]
    active_asset = "START"
    applied_regimes = [np.nan]
    target_assets = ["START"]
    strategy_returns = [0.0]
    switch_costs = [0.0]
    position_changed = [False]
    for idx in range(1, len(data)):
        regime = int(data["predicted_regime"].iloc[idx - 1])
        target = REGIME_TO_ASSET.get(regime, "BIL")
        spy_ret = data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0
        qqq_ret = data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0
        bil_ret = data["defensive_close"].iloc[idx] / data["defensive_close"].iloc[idx - 1] - 1.0 if pd.notna(data["defensive_close"].iloc[idx - 1]) and pd.notna(data["defensive_close"].iloc[idx]) else 0.0
        gross = qqq_ret if target == "QQQ" else spy_ret if target == "SPY" else bil_ret
        changed = target != active_asset
        cost = SWITCH_COST if changed else 0.0
        strat_ret = gross - cost
        equity.append(equity[-1] * (1.0 + strat_ret))
        spy_equity.append(spy_equity[-1] * (1.0 + spy_ret))
        applied_regimes.append(regime)
        target_assets.append(target)
        strategy_returns.append(strat_ret)
        switch_costs.append(cost)
        position_changed.append(changed)
        active_asset = target
    out = data.copy()
    out["variant"] = variant
    out["applied_regime"] = applied_regimes
    out["target_asset"] = target_assets
    out["switch_cost"] = switch_costs
    out["strategy_return"] = strategy_returns
    out["position_changed"] = position_changed
    out["strategy_equity"] = equity
    out["spy_equity"] = spy_equity
    out["drawdown"] = out["strategy_equity"] / out["strategy_equity"].cummax() - 1.0
    out["report_date"] = pd.to_datetime(out["timestamp"]).dt.date.astype(str)
    daily = out.loc[out.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    daily["spy_daily_return"] = daily["spy_equity"].pct_change().fillna(0.0)
    metrics = metrics_from_daily(daily, variant)
    metrics["Switch Count"] = int(out["position_changed"].astype(bool).sum())
    metrics["daily_rows"] = len(daily)
    metrics["hourly_price_update_rows"] = len(out)
    return out, daily, metrics


def yearly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in daily.groupby(pd.to_datetime(daily["timestamp"]).dt.year):
        rows.append(
            {
                "year": int(year),
                "strategy_return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "start_equity": float(group["strategy_equity"].iloc[0]),
                "end_equity": float(group["strategy_equity"].iloc[-1]),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def signal_match(candidate: pd.DataFrame, reference: pd.DataFrame) -> float:
    cmp = candidate[["timestamp", "predicted_regime"]].merge(
        reference[["timestamp", "predicted_regime"]], on="timestamp", how="inner", suffixes=("_candidate", "_reference")
    )
    if cmp.empty:
        return np.nan
    return float((cmp["predicted_regime_candidate"].astype(int) == cmp["predicted_regime_reference"].astype(int)).mean())


def label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in LABEL_SPECS:
        vc = labels[spec.name].dropna().astype(int).value_counts().sort_index()
        total = int(vc.sum())
        for regime in [0, 1, 2]:
            count = int(vc.get(regime, 0))
            rows.append(
                {
                    "label_family": spec.name,
                    "family": spec.family,
                    "regime": regime,
                    "regime_name": REGIME_NAME[regime],
                    "count": count,
                    "share": count / total if total else np.nan,
                    "params": json.dumps(spec.params, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def write_readme(best: dict[str, Any], any_beats_spy: bool, material_close: bool) -> None:
    lines = [
        "# MUB-27 V4.2 Strict-PIT Label Redesign",
        "",
        "This is isolated research output only.",
        "",
        "## Governance",
        "",
        "- Labels use future outcomes only as pre-declared supervised targets.",
        "- No full-sample teacher labels are used for training.",
        "- Scaling, XGBoost fitting, attention training, and early stopping are fold-local and train-period only.",
        "- Allocation is unlevered: Expansion QQQ, Slowdown SPY, Recession BIL.",
        "- No production V4 Delta tables, Saxo execution, token refresh, RT/dashboard notebooks, live schedules, model registry entries, or trading logic were modified.",
        "",
        "## Conclusion",
        "",
        f"- Best strict-PIT variant: `{best.get('variant')}`",
        f"- Any strict-PIT variant beats SPY: `{any_beats_spy}`",
        f"- Materially closes B/C2 gap: `{material_close}`",
        "",
    ]
    write_text("\n".join(lines), OUTPUT_ROOT / "README.md")


def main() -> None:
    ensure_dirs()
    seed_everything()
    grid, labels = load_grid_and_labels()
    bil_prices = load_bil_prices()
    b_c2_ref, raw_ref = load_reference_signals()
    for spec in LABEL_SPECS:
        write_csv(labels[["timestamp", "label_available_ts", "fwd_return_1w", "trailing_vol_1w", "vol_adj_score", "future_max_drawdown_1w", spec.name]], LABELS_DIR / f"{spec.name}_labels.csv")

    all_metrics, all_signals, all_fold_manifests = [], [], []
    for spec in LABEL_SPECS:
        signals, manifest = run_xgboost_label(grid, spec.name, raw_ref)
        variant = f"xgboost_{spec.name}"
        write_csv(signals, SIGNALS_DIR / f"{variant}_signals.csv")
        hourly, daily, metrics = run_backtest(grid, signals, variant, bil_prices)
        write_csv(daily, PERFORMANCE_DIR / f"{variant}_daily_returns.csv")
        write_csv(daily[["timestamp", "report_date", "strategy_equity", "spy_equity"]], PERFORMANCE_DIR / f"{variant}_equity_curve_daily.csv")
        write_csv(yearly_returns(daily), PERFORMANCE_DIR / f"{variant}_yearly_returns.csv")
        metrics.update(
            {
                "model_family": "xgboost",
                "label_family": spec.name,
                "label_rule_family": spec.family,
                "Match vs B/C2": signal_match(signals, b_c2_ref),
                "Match vs raw-label WF": signal_match(signals, raw_ref),
            }
        )
        all_metrics.append(metrics)
        all_signals.append(signals)
        all_fold_manifests.append(manifest)

    xgb_summary = pd.DataFrame(all_metrics).sort_values(["CAGR", "Sharpe"], ascending=False)
    top_labels = xgb_summary["label_family"].head(2).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for label_col in top_labels:
        signals, manifest = run_tft_label(grid, label_col, raw_ref, device)
        variant = f"tft72_bil_{label_col}"
        write_csv(signals, SIGNALS_DIR / f"{variant}_signals.csv")
        hourly, daily, metrics = run_backtest(grid, signals, variant, bil_prices)
        write_csv(daily, PERFORMANCE_DIR / f"{variant}_daily_returns.csv")
        write_csv(daily[["timestamp", "report_date", "strategy_equity", "spy_equity"]], PERFORMANCE_DIR / f"{variant}_equity_curve_daily.csv")
        write_csv(yearly_returns(daily), PERFORMANCE_DIR / f"{variant}_yearly_returns.csv")
        metrics.update(
            {
                "model_family": "tft_attention_72h",
                "label_family": label_col,
                "label_rule_family": next(s.family for s in LABEL_SPECS if s.name == label_col),
                "Match vs B/C2": signal_match(signals, b_c2_ref),
                "Match vs raw-label WF": signal_match(signals, raw_ref),
            }
        )
        all_metrics.append(metrics)
        all_signals.append(signals)
        all_fold_manifests.append(manifest)

    comparison = pd.DataFrame(all_metrics)
    comparison["Gap vs SPY CAGR"] = comparison["CAGR"] - SPY_CAGR
    comparison["Gap vs B/C2 CAGR"] = comparison["CAGR"] - B_C2_CAGR
    comparison["pit_status"] = "strict_pit"
    comparison = comparison.sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)

    label_dist = label_distribution(labels)
    write_csv(label_dist, FORENSICS_DIR / "label_distribution_summary.csv")
    write_csv(comparison, FORENSICS_DIR / "variant_comparison.csv")
    write_csv(comparison[["variant", "model_family", "label_family", "CAGR", "Gap vs SPY CAGR", "Gap vs B/C2 CAGR", "Max DD", "Final Equity"]], FORENSICS_DIR / "gap_vs_spy_and_b_c2.csv")
    write_csv(pd.concat(all_fold_manifests, ignore_index=True), FORENSICS_DIR / "fold_manifest.csv")
    write_csv(pd.concat(all_signals, ignore_index=True), FORENSICS_DIR / "all_label_redesign_signals.csv")

    best = comparison.iloc[0].to_dict()
    any_beats_spy = bool((comparison["CAGR"] > SPY_CAGR).any())
    material_close = bool((comparison["Gap vs B/C2 CAGR"] > -0.05).any())
    write_readme(best, any_beats_spy, material_close)

    input_hashes = {
        "faithful_legacy_grid": {"path": FAITHFUL_GRID_PATH, "sha256": file_sha256(FAITHFUL_GRID_PATH)},
        "defensive_etf_hourly_prices": {"path": DEFENSIVE_PRICE_PATH, "sha256": file_sha256(DEFENSIVE_PRICE_PATH)},
        "prior_strict_pit_root": str(STRICT_ROOT),
        "prior_bilstm_root": str(BILSTM_ROOT),
        "prior_tft_root": str(TFT_ROOT),
        "prior_tft_window_defensive_root": str(WINDOW_ROOT),
        "prior_leverage_root": str(LEVERAGE_ROOT),
        "b_c2_reference_signals": {"path": B_C2_SIGNAL_PATH, "sha256": file_sha256(B_C2_SIGNAL_PATH)},
        "raw_wf_reference_signals": {"path": RAW_WF_SIGNAL_PATH, "sha256": file_sha256(RAW_WF_SIGNAL_PATH)},
        "window_comparison": {"path": WINDOW_COMPARISON_PATH, "sha256": file_sha256(WINDOW_COMPARISON_PATH)},
    }
    write_json(input_hashes, MANIFESTS_DIR / "input_hashes.json")
    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "xgboost_version": xgb.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "device": str(device),
        "output_root": OUTPUT_ROOT,
        "label_specs": [{"name": s.name, "family": s.family, "params": s.params} for s in LABEL_SPECS],
        "top_labels_retested_with_tft": top_labels,
        "allocation": REGIME_TO_ASSET,
        "switch_cost_bps": 5,
        "strict_pit_controls": [
            "monthly walk-forward folds",
            "training labels require label_available_ts strictly before fold_start",
            "scaler fit on train period only",
            "no full-sample teacher labels for training",
            "TFT early stopping uses train-period validation only",
            "unlevered allocation only",
        ],
        "production_delta_writes": False,
        "workflow_or_schedule_changes": False,
        "saxo_or_token_changes": False,
        "rt_or_dashboard_changes": False,
        "model_registry_writes": False,
        "trading_logic_changes": False,
        "best_variant": best,
        "any_strict_pit_variant_beats_spy": any_beats_spy,
        "materially_closes_b_c2_gap": material_close,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")
    print(comparison.to_string(index=False))
    print(json.dumps(jsonable({"best": best, "top_labels_retested_with_tft": top_labels}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
