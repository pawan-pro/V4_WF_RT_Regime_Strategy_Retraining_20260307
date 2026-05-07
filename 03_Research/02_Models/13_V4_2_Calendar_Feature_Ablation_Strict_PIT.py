# Databricks notebook source
"""MUB-27 V4.2 strict-PIT calendar/time feature ablation.

Research-only. This script writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_calendar_feature_ablation_2026-05-07/`

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
OUTPUT_ROOT = BASE / "v4_2_calendar_feature_ablation_2026-05-07"
FEATURES_DIR = OUTPUT_ROOT / "features"
SIGNALS_DIR = OUTPUT_ROOT / "signals"
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"
CHARTS_DIR = OUTPUT_ROOT / "charts"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"

FAITHFUL_GRID_PATH = (
    BASE / "v4_1_csv" / "full_legacy_grid_clone_2026-04-30" / "grids" / "full_legacy_grid_clone.csv"
)
DEFENSIVE_PRICE_ROOT = BASE / "v4_2_defensive_sleeve_prices_2026-05-06"
DEFENSIVE_PRICE_PATH = DEFENSIVE_PRICE_ROOT / "prices" / "defensive_etf_hourly_prices.csv"
LABEL_ROOT = BASE / "v4_2_label_redesign_strict_pit_2026-05-06"
LABEL_PATH = LABEL_ROOT / "labels" / "drawdown_aware_dd_010_ret_005_labels.csv"
LABEL_COMPARISON_PATH = LABEL_ROOT / "forensics" / "variant_comparison.csv"
WINDOW_ROOT = BASE / "v4_2_tft_window_defensive_sleeve_2026-05-06"
LEVERAGE_ROOT = BASE / "v4_2_leverage_sensitivity_2026-05-06"
STRICT_ROOT = BASE / "v4_2_strict_pit_research_2026-05-06"
B_C2_SIGNAL_PATH = STRICT_ROOT / "signals" / "B_reference_teacher_smoothed_signals.csv"

TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FINAL_FOLD_END = pd.Timestamp("2026-01-01")
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
PRIOR_BEST = {
    "variant": "tft72_bil_drawdown_aware_dd_010_ret_005",
    "CAGR": 0.110980,
    "Sharpe": 1.0233,
    "Max DD": -0.156828,
    "Final Equity": 8_462_552.454,
}
REGIME_NAME = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "BIL"}
LABEL_COL = "drawdown_aware_dd_010_ret_005"

BASE_FEATURES = [
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

DOW_FEATURES = ["is_monday", "is_tuesday", "is_wednesday", "is_thursday", "is_friday", "is_weekend"]
WEEK_MOD3_FEATURES = ["week_mod3_0", "week_mod3_1", "week_mod3_2"]
SESSION_FEATURES = [
    "hour_sin",
    "hour_cos",
    "is_market_open_hour",
    "is_market_close_hour",
    "is_regular_trading_hours",
    "is_month_end",
    "is_quarter_end",
    "is_first_trading_day_month",
    "is_last_trading_day_month",
]

FEATURE_SETS = {
    "A_baseline_14": BASE_FEATURES,
    "B_day_of_week": BASE_FEATURES + DOW_FEATURES,
    "C_day_of_week_week_mod3": BASE_FEATURES + DOW_FEATURES + WEEK_MOD3_FEATURES,
    "D_calendar_session": BASE_FEATURES + DOW_FEATURES + WEEK_MOD3_FEATURES + SESSION_FEATURES,
}


def ensure_python_deps():
    missing = []
    for module_name, package_name in [("torch", "torch"), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib")]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    import sklearn
    import torch
    from sklearn.preprocessing import StandardScaler

    return torch, sklearn, StandardScaler


torch, sklearn, StandardScaler = ensure_python_deps()
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RunResult:
    feature_set: str
    signals: pd.DataFrame
    fold_manifest: pd.DataFrame


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
    for path in [FEATURES_DIR, SIGNALS_DIR, PERFORMANCE_DIR, FORENSICS_DIR, CHARTS_DIR, MANIFESTS_DIR]:
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


def dist(series: pd.Series | np.ndarray) -> dict[str, int]:
    if series is None or len(series) == 0:
        return {}
    vc = pd.Series(series).dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


def add_calendar_features(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    ts = pd.to_datetime(out["timestamp"])
    dow = ts.dt.dayofweek
    out["is_monday"] = (dow == 0).astype(float)
    out["is_tuesday"] = (dow == 1).astype(float)
    out["is_wednesday"] = (dow == 2).astype(float)
    out["is_thursday"] = (dow == 3).astype(float)
    out["is_friday"] = (dow == 4).astype(float)
    out["is_weekend"] = (dow >= 5).astype(float)
    week_mod3 = ts.dt.isocalendar().week.astype(int) % 3
    for idx in [0, 1, 2]:
        out[f"week_mod3_{idx}"] = (week_mod3 == idx).astype(float)
    hour_float = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    out["is_market_open_hour"] = ((ts.dt.hour == 14) | (ts.dt.hour == 15)).astype(float)
    out["is_market_close_hour"] = ((ts.dt.hour == 20) | (ts.dt.hour == 21)).astype(float)
    out["is_regular_trading_hours"] = ((ts.dt.hour >= 14) & (ts.dt.hour <= 21) & (dow < 5)).astype(float)
    out["is_month_end"] = ts.dt.is_month_end.astype(float)
    out["is_quarter_end"] = ts.dt.is_quarter_end.astype(float)
    date_key = ts.dt.date.astype(str)
    rth_rows = out[out["is_regular_trading_hours"].eq(1.0)][["timestamp"]].copy()
    rth_rows["month"] = rth_rows["timestamp"].dt.to_period("M").astype(str)
    rth_rows["date_key"] = rth_rows["timestamp"].dt.date.astype(str)
    month_first = rth_rows.groupby("month")["date_key"].min()
    month_last = rth_rows.groupby("month")["date_key"].max()
    month_key = ts.dt.to_period("M").astype(str)
    out["is_first_trading_day_month"] = [float(d == month_first.get(m, "")) for d, m in zip(date_key, month_key)]
    out["is_last_trading_day_month"] = [float(d == month_last.get(m, "")) for d, m in zip(date_key, month_key)]
    return out


def load_grid() -> pd.DataFrame:
    grid = read_csv(FAITHFUL_GRID_PATH)
    label = read_csv(LABEL_PATH)
    cols = ["timestamp", "SPY_close", "QQQ_close", *BASE_FEATURES]
    missing = [c for c in cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Faithful grid missing required columns: {missing}")
    grid = grid[cols].copy()
    for col in ["SPY_close", "QQQ_close", *BASE_FEATURES]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    label_cols = ["timestamp", "label_available_ts", LABEL_COL]
    missing_label = [c for c in label_cols if c not in label.columns]
    if missing_label:
        raise ValueError(f"Label redesign artifact missing columns: {missing_label}")
    merged = grid.merge(label[label_cols], on="timestamp", how="left")
    merged[LABEL_COL] = pd.to_numeric(merged[LABEL_COL], errors="coerce")
    merged["label_available_ts"] = pd.to_datetime(merged["label_available_ts"], errors="coerce")
    merged = add_calendar_features(merged)
    all_possible_features = sorted(set(BASE_FEATURES + DOW_FEATURES + WEEK_MOD3_FEATURES + SESSION_FEATURES))
    merged["feature_ok"] = merged[all_possible_features + ["SPY_close", "QQQ_close"]].notna().all(axis=1)
    return merged


def working_frame(grid: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    need = ["timestamp", "SPY_close", "QQQ_close", "feature_ok", "label_available_ts", LABEL_COL, *features]
    out = grid[need].copy()
    out = out[(out["timestamp"] >= TRAIN_START) & out["feature_ok"] & out[LABEL_COL].notna()].copy()
    out[LABEL_COL] = out[LABEL_COL].astype(int)
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


def load_b_c2_reference() -> pd.DataFrame:
    ref = read_csv(B_C2_SIGNAL_PATH)
    ref["predicted_regime"] = pd.to_numeric(ref["predicted_regime"], errors="coerce").astype("Int64")
    ref = ref.dropna(subset=["predicted_regime"]).copy()
    ref["predicted_regime"] = ref["predicted_regime"].astype(int)
    return ref


def canonical_eval(work: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    ref_ts = set(ref["timestamp"])
    out = work[(work["timestamp"] >= EVAL_START) & (work["timestamp"] <= EVAL_END) & work["timestamp"].isin(ref_ts)].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


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
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
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


def run_feature_set(grid: pd.DataFrame, feature_set: str, features: list[str], ref: pd.DataFrame, device) -> RunResult:
    work = working_frame(grid, features)
    canonical = canonical_eval(work, ref)
    all_features = work[features].to_numpy(dtype=np.float32)
    labels = work[LABEL_COL].to_numpy(dtype=int)
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
        sig = test_df[["timestamp", LABEL_COL, "predicted_regime"]].copy().rename(columns={LABEL_COL: "label_regime"})
        sig["variant"] = feature_set
        sig["feature_set"] = feature_set
        sig["model_family"] = "tft_attention_72h"
        sig["label_family"] = LABEL_COL
        sig["sequence_window_hours"] = SEQUENCE_WINDOW
        sig["regime_name"] = sig["predicted_regime"].map(REGIME_NAME)
        sig["allocation_scale"] = 1.0
        rows.append(sig)
        manifests.append(
            {
                "feature_set": feature_set,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "feature_count": len(features),
                "train_sequence_count_before_sampling": len(train_pos),
                "train_sequence_count_after_sampling": len(train_pos_sample),
                "validation_sequence_count": len(val_pos),
                "test_sequence_count": len(test_df),
                "train_label_distribution": dist(labels[train_pos]),
                "prediction_distribution": dist(test_df["predicted_regime"]),
                "strict_pit": True,
                "feature_hash": hashlib.sha256(",".join(features).encode("utf-8")).hexdigest(),
                "train_input_hash": frame_hash(work.iloc[train_pos_sample][["timestamp", LABEL_COL, "label_available_ts", *features]]),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return RunResult(feature_set, pd.concat(rows, ignore_index=True), pd.DataFrame(manifests))


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


def metrics_from_daily(daily: pd.DataFrame, feature_set: str) -> dict[str, Any]:
    ret = daily["strategy_equity"].astype(float).pct_change().dropna()
    downside = ret[ret < 0]
    final_equity = float(daily["strategy_equity"].iloc[-1])
    dd = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1.0
    vol = ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0
    max_dd = float(dd.min())
    cagr = (final_equity / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0
    return {
        "feature_set": feature_set,
        "variant": feature_set,
        "CAGR": cagr,
        "Sharpe": ret.mean() / ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0,
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0) if len(downside) > 1 and downside.std(ddof=1) else 0.0,
        "Max DD": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "Final Equity": final_equity,
        "Annualized Volatility": vol,
    }


def run_backtest(grid: pd.DataFrame, signals: pd.DataFrame, feature_set: str, bil_prices: pd.DataFrame):
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
    out["feature_set"] = feature_set
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
    metrics = metrics_from_daily(daily, feature_set)
    metrics["Switch Count"] = int(out["position_changed"].astype(bool).sum())
    metrics["daily_rows"] = len(daily)
    metrics["hourly_price_update_rows"] = len(out)
    return out, daily, metrics


def yearly_returns(daily: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    rows = []
    for year, group in daily.groupby(pd.to_datetime(daily["timestamp"]).dt.year):
        rows.append(
            {
                "feature_set": feature_set,
                "year": int(year),
                "strategy_return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "spy_return": float(group["spy_equity"].iloc[-1] / group["spy_equity"].iloc[0] - 1.0),
                "start_equity": float(group["strategy_equity"].iloc[0]),
                "end_equity": float(group["strategy_equity"].iloc[-1]),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def monthly_returns(daily: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    month_key = pd.to_datetime(daily["timestamp"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in daily.groupby(month_key):
        rows.append(
            {
                "feature_set": feature_set,
                "month": month,
                "strategy_return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "spy_return": float(group["spy_equity"].iloc[-1] / group["spy_equity"].iloc[0] - 1.0),
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


def regime_distribution(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set, group in signals.groupby("feature_set"):
        vc = group["predicted_regime"].astype(int).value_counts().sort_index()
        total = int(vc.sum())
        for regime in [0, 1, 2]:
            count = int(vc.get(regime, 0))
            rows.append(
                {
                    "feature_set": feature_set,
                    "regime": regime,
                    "regime_name": REGIME_NAME[regime],
                    "count": count,
                    "share": count / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def feature_inventory() -> pd.DataFrame:
    rows = []
    base_set = set(BASE_FEATURES)
    for feature_set, features in FEATURE_SETS.items():
        for feature in features:
            if feature in BASE_FEATURES:
                category = "original_v4"
            elif feature in DOW_FEATURES:
                category = "day_of_week"
            elif feature in WEEK_MOD3_FEATURES:
                category = "week_mod3"
            else:
                category = "session_calendar"
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature": feature,
                    "category": category,
                    "is_calendar_time_feature": feature not in base_set,
                    "known_in_advance": feature not in base_set,
                    "raw_year_feature": feature.lower() in {"year", "raw_year"},
                }
            )
    return pd.DataFrame(rows)


def plot_cagr_drawdown(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary["Max DD"], summary["CAGR"], s=90)
    for _, row in summary.iterrows():
        ax.annotate(row["feature_set"], (row["Max DD"], row["CAGR"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(SPY_CAGR, color="black", linestyle="--", linewidth=1.0, label="SPY CAGR")
    ax.axhline(B_C2_CAGR, color="tab:red", linestyle=":", linewidth=1.0, label="B/C2 CAGR")
    ax.set_title("Calendar Feature Ablation: CAGR vs Max Drawdown")
    ax.set_xlabel("Max Drawdown")
    ax.set_ylabel("CAGR")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = CHARTS_DIR / "cagr_vs_maxdd_feature_sets.png"
    assert_output_path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_yearly(best_set: str, yearly: pd.DataFrame) -> None:
    subset = yearly[yearly["feature_set"].isin([best_set, "A_baseline_14"])].copy()
    pivot = subset.pivot(index="year", columns="feature_set", values="strategy_return")
    spy = subset.groupby("year")["spy_return"].first()
    pivot["SPY"] = spy
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Yearly Returns: Best vs Baseline vs SPY")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.grid(True, axis="y", alpha=0.25)
    path = CHARTS_DIR / "yearly_returns_best_vs_baseline_spy.png"
    assert_output_path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_readme(summary: pd.DataFrame, baseline_recon: dict[str, Any], selections: dict[str, Any]) -> None:
    lines = [
        "# MUB-27 V4.2 Calendar Feature Ablation",
        "",
        "Research-only strict-PIT TFT 72h feature ablation.",
        "",
        "## Governance",
        "",
        "- Calendar/time features are known in advance.",
        "- No raw year feature is included.",
        "- Model architecture, label, BIL sleeve, leverage, switch cost, and validation remain fixed across feature sets.",
        "- No production V4 Delta tables, Saxo execution, token refresh, RT/dashboard notebooks, live schedules, model registry entries, or trading logic were modified.",
        "",
        "## Baseline Reconciliation",
        "",
        json.dumps(jsonable(baseline_recon), indent=2, sort_keys=True),
        "",
        "## Selections",
        "",
        json.dumps(jsonable(selections), indent=2, sort_keys=True),
        "",
        "## Summary",
        "",
        summary.to_csv(index=False),
    ]
    write_text("\n".join(lines), OUTPUT_ROOT / "README.md")


def yearly_stability_score(yearly: pd.DataFrame, feature_set: str) -> dict[str, Any]:
    rows = yearly[yearly["feature_set"].eq(feature_set)]["strategy_return"].astype(float)
    return {
        "yearly_return_std": float(rows.std(ddof=1)) if len(rows) > 1 else np.nan,
        "negative_year_count": int((rows < 0).sum()),
        "worst_year": float(rows.min()) if len(rows) else np.nan,
    }


def select_balanced(summary: pd.DataFrame, yearly: pd.DataFrame, regimes: pd.DataFrame) -> dict[str, Any]:
    baseline = summary[summary["feature_set"].eq("A_baseline_14")].iloc[0]
    candidates = summary[
        (summary["CAGR"] > baseline["CAGR"])
        & (summary["Sharpe"] >= baseline["Sharpe"] - 1e-9)
        & (summary["Max DD"] >= baseline["Max DD"] - 1e-9)
    ].copy()
    stable = []
    for row in candidates.itertuples(index=False):
        dist_df = regimes[regimes["feature_set"].eq(row.feature_set)]
        max_share = float(dist_df["share"].max()) if len(dist_df) else 1.0
        stability = yearly_stability_score(yearly, row.feature_set)
        stable.append({**row._asdict(), **stability, "max_regime_share": max_share, "degenerate_regime_distribution": max_share > 0.90})
    stable_df = pd.DataFrame(stable)
    if stable_df.empty:
        return {"feature_set": None, "reason": "no feature set improved CAGR while preserving Sharpe and Max DD"}
    stable_df = stable_df[~stable_df["degenerate_regime_distribution"]].copy()
    if stable_df.empty:
        return {"feature_set": None, "reason": "only qualifying candidates had degenerate regime distributions"}
    stable_df = stable_df.sort_values(["CAGR", "Sharpe", "Max DD"], ascending=False)
    return stable_df.iloc[0].to_dict()


def main() -> None:
    ensure_dirs()
    seed_everything()
    grid = load_grid()
    bil_prices = load_bil_prices()
    b_c2_ref = load_b_c2_reference()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inv = feature_inventory()
    write_csv(inv, FEATURES_DIR / "feature_set_inventory.csv")
    if inv["raw_year_feature"].any():
        raise ValueError("Raw year feature detected in feature inventory")

    all_metrics, all_signals, all_folds, all_yearly, all_monthly = [], [], [], [], []
    for feature_set, features in FEATURE_SETS.items():
        result = run_feature_set(grid, feature_set, features, b_c2_ref, device)
        write_csv(result.signals, SIGNALS_DIR / f"{feature_set}_signals.csv")
        hourly, daily, metrics = run_backtest(grid, result.signals, feature_set, bil_prices)
        write_csv(daily, PERFORMANCE_DIR / f"{feature_set}_daily_returns.csv")
        write_csv(daily[["timestamp", "report_date", "strategy_equity", "spy_equity"]], PERFORMANCE_DIR / f"{feature_set}_equity_curve_daily.csv")
        yr = yearly_returns(daily, feature_set)
        mo = monthly_returns(daily, feature_set)
        write_csv(yr, PERFORMANCE_DIR / f"{feature_set}_yearly_returns.csv")
        all_yearly.append(yr)
        all_monthly.append(mo)
        metrics.update(
            {
                "feature_count": len(features),
                "calendar_feature_count": len([f for f in features if f not in BASE_FEATURES]),
                "Match vs B/C2": signal_match(result.signals, b_c2_ref),
                "Gap vs SPY CAGR": metrics["CAGR"] - SPY_CAGR,
                "Gap vs B/C2 CAGR": metrics["CAGR"] - B_C2_CAGR,
                "raw_year_feature_included": False,
            }
        )
        all_metrics.append(metrics)
        all_signals.append(result.signals)
        all_folds.append(result.fold_manifest)

    summary = pd.DataFrame(all_metrics).sort_values(["CAGR", "Sharpe"], ascending=False).reset_index(drop=True)
    signals_all = pd.concat(all_signals, ignore_index=True)
    yearly_all = pd.concat(all_yearly, ignore_index=True)
    monthly_all = pd.concat(all_monthly, ignore_index=True)
    regimes = regime_distribution(signals_all)
    yearly_std = yearly_all.groupby("feature_set")["strategy_return"].std(ddof=1).rename("yearly_return_std").reset_index()
    summary = summary.merge(yearly_std, on="feature_set", how="left")

    write_csv(summary, FORENSICS_DIR / "calendar_feature_ablation_summary.csv")
    write_csv(yearly_all, FORENSICS_DIR / "yearly_return_comparison.csv")
    write_csv(monthly_all, FORENSICS_DIR / "monthly_return_comparison.csv")
    write_csv(regimes, FORENSICS_DIR / "regime_distribution_by_feature_set.csv")
    write_csv(pd.concat(all_folds, ignore_index=True), FORENSICS_DIR / "fold_manifest.csv")

    baseline = summary[summary["feature_set"].eq("A_baseline_14")].iloc[0]
    baseline_recon = {
        "feature_set": "A_baseline_14",
        "observed_CAGR": baseline["CAGR"],
        "observed_Sharpe": baseline["Sharpe"],
        "observed_Max_DD": baseline["Max DD"],
        "prior_CAGR": PRIOR_BEST["CAGR"],
        "prior_Sharpe": PRIOR_BEST["Sharpe"],
        "prior_Max_DD": PRIOR_BEST["Max DD"],
        "reconciles_close": bool(
            abs(baseline["CAGR"] - PRIOR_BEST["CAGR"]) < 0.002
            and abs(baseline["Sharpe"] - PRIOR_BEST["Sharpe"]) < 0.03
            and abs(baseline["Max DD"] - PRIOR_BEST["Max DD"]) < 0.01
        ),
        "note": "Exact equality can differ because this reruns deterministic PyTorch fold training in a new script/runtime; architecture, label, feature set, and performance methodology are held fixed.",
    }
    selections = {
        "best_cagr_feature_set": summary.sort_values(["CAGR", "Sharpe"], ascending=False).iloc[0].to_dict(),
        "best_sharpe_feature_set": summary.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0].to_dict(),
        "best_calmar_feature_set": summary.sort_values(["Calmar", "CAGR"], ascending=False).iloc[0].to_dict(),
        "best_balanced_feature_set": select_balanced(summary, yearly_all, regimes),
        "any_feature_set_beats_spy_cagr": bool((summary["CAGR"] > SPY_CAGR).any()),
        "calendar_features_improve_yearly_stability": bool(
            summary[~summary["feature_set"].eq("A_baseline_14")]["yearly_return_std"].min()
            < summary[summary["feature_set"].eq("A_baseline_14")]["yearly_return_std"].iloc[0]
        ),
    }

    plot_cagr_drawdown(summary)
    plot_yearly(selections["best_cagr_feature_set"]["feature_set"], yearly_all)
    write_readme(summary, baseline_recon, selections)

    input_hashes = {
        "faithful_legacy_grid": {"path": FAITHFUL_GRID_PATH, "sha256": file_sha256(FAITHFUL_GRID_PATH)},
        "defensive_etf_hourly_prices": {"path": DEFENSIVE_PRICE_PATH, "sha256": file_sha256(DEFENSIVE_PRICE_PATH)},
        "label_redesign_label_file": {"path": LABEL_PATH, "sha256": file_sha256(LABEL_PATH)},
        "label_redesign_comparison": {"path": LABEL_COMPARISON_PATH, "sha256": file_sha256(LABEL_COMPARISON_PATH)},
        "prior_tft_window_defensive_root": str(WINDOW_ROOT),
        "prior_leverage_root": str(LEVERAGE_ROOT),
        "b_c2_reference_signals": {"path": B_C2_SIGNAL_PATH, "sha256": file_sha256(B_C2_SIGNAL_PATH)},
    }
    write_json(input_hashes, MANIFESTS_DIR / "input_hashes.json")
    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
        "device": str(device),
        "output_root": OUTPUT_ROOT,
        "feature_sets": FEATURE_SETS,
        "raw_year_feature_included": False,
        "model": "TFT_attention_72h",
        "label": LABEL_COL,
        "defensive_sleeve": "BIL",
        "allocation": REGIME_TO_ASSET,
        "switch_cost_bps": 5,
        "strict_pit_controls": [
            "monthly walk-forward folds",
            "calendar/time features only use known-in-advance timestamp information",
            "training labels require label_available_ts strictly before fold_start",
            "scaler fit on train period only",
            "sequence construction uses rows up to prediction timestamp only",
            "no full-sample teacher labels for training",
            "B/C2 used for comparison only",
            "unlevered allocation only",
        ],
        "baseline_reconciliation": baseline_recon,
        "selection": selections,
        "production_delta_writes": False,
        "workflow_or_schedule_changes": False,
        "saxo_or_token_changes": False,
        "rt_or_dashboard_changes": False,
        "model_registry_writes": False,
        "trading_logic_changes": False,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")
    print(summary.to_string(index=False))
    print(json.dumps(jsonable({"baseline_reconciliation": baseline_recon, "selection": selections}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
