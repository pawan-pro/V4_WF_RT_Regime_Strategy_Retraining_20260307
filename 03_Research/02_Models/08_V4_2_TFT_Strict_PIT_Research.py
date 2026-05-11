# Databricks notebook source
"""MUB-27 V4.2 TFT-style strict-PIT sequence research.

Research-only. Reads faithful legacy CSV artifacts plus prior MUB-27 research
outputs and writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_strict_pit_2026-05-06/`

No production Delta writes, Databricks job/workflow mutations, Saxo/token
changes, RT/dashboard notebook edits, live schedule changes, model registry
writes, or trading-logic edits are performed.
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
OUTPUT_ROOT = BASE / "v4_2_tft_strict_pit_2026-05-06"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
SIGNALS_DIR = OUTPUT_ROOT / "signals"
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"

FAITHFUL_GRID_PATH = (
    BASE
    / "v4_1_csv"
    / "full_legacy_grid_clone_2026-04-30"
    / "grids"
    / "full_legacy_grid_clone.csv"
)
REFERENCE_WALKFORWARD_PATH = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
    / "walkforward_signals.csv"
)
REFERENCE_FULL_SIGNAL_PATH = (
    BASE
    / "comparison"
    / "2026-04-20_canonical_legacy_v4_baseline_vs_corrected"
    / "corrected_variant"
    / "full_signal_export.csv"
)
MUB27_STRICT_ROOT = BASE / "v4_2_strict_pit_research_2026-05-06"
MUB27_COMPARISON_PATH = MUB27_STRICT_ROOT / "forensics" / "variant_comparison.csv"
MUB27_SIGNAL_DIR = MUB27_STRICT_ROOT / "signals"
BILSTM_ROOT = BASE / "v4_2_bilstm_strict_pit_2026-05-06"
BILSTM_COMPARISON_PATH = BILSTM_ROOT / "forensics" / "comparison_vs_A_B_C_D_SPY.csv"
BILSTM_SIGNAL_PATH = BILSTM_ROOT / "signals" / "bilstm_signals.csv"

TRAIN_START = pd.Timestamp("2016-01-01")
EVAL_START = pd.Timestamp("2021-01-01")
EVAL_END = pd.Timestamp("2025-12-31 23:59:59")
FINAL_FOLD_END = pd.Timestamp("2026-01-01")
FWD_HOURS = 168
INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005

SEQUENCE_WINDOWS = [24, 72]
MAX_EPOCHS = 6
PATIENCE = 1
BATCH_SIZE = 512
MAX_TRAIN_SAMPLES_PER_CLASS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

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

REGIME_NAME = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "CASH"}
BASELINE_SIGNAL_FILES = {
    "A_raw_label_walkforward": MUB27_SIGNAL_DIR / "A_raw_label_walkforward_signals.csv",
    "B_reference_teacher_smoothed": MUB27_SIGNAL_DIR / "B_reference_teacher_smoothed_signals.csv",
    "C_strict_rolling_teacher_student": MUB27_SIGNAL_DIR / "C_strict_rolling_teacher_student_signals.csv",
    "D_pca_strict_xgboost": MUB27_SIGNAL_DIR / "D_pca_strict_xgboost_signals.csv",
    "E_spy_benchmark": MUB27_SIGNAL_DIR / "E_spy_benchmark_signals.csv",
}


def ensure_python_deps():
    missing = []
    for module_name, package_name in [("torch", "torch"), ("sklearn", "scikit-learn")]:
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    for path in [MANIFESTS_DIR, SIGNALS_DIR, PERFORMANCE_DIR, FORENSICS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_research_output_path(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside research output root: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_research_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_text(text: str, path: Path) -> None:
    assert_research_output_path(path)
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
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(payload: dict[str, Any], path: Path) -> None:
    assert_research_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    out = df.copy().astype(object).where(pd.notna(df), "")
    cols = [str(c) for c in out.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    use = df[cols].copy() if cols else df.copy()
    return hashlib.sha256(use.sort_index(axis=1).to_csv(index=False).encode("utf-8")).hexdigest()


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        raise ValueError("Expected timestamp column")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    try:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    except Exception:
        pass
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return normalize_timestamp(pd.read_csv(path, **kwargs))


def label_regime(ret: float) -> float:
    if pd.isna(ret):
        return np.nan
    if ret > 0.005:
        return 0
    if ret < -0.005:
        return 2
    return 1


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


class TemporalAttentionClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=gru_dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x):
        projected = torch.tanh(self.input_projection(x))
        encoded, _ = self.encoder(projected)
        scores = self.attention(encoded).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(encoded * weights, dim=1)
        return self.head(self.dropout(context))


def load_grid() -> pd.DataFrame:
    grid = read_csv(FAITHFUL_GRID_PATH)
    cols = ["timestamp", "SPY_close", "QQQ_close", *FEATURES]
    missing = [c for c in cols if c not in grid.columns]
    if missing:
        raise ValueError(f"Faithful grid missing required columns: {missing}")
    grid = grid[cols].copy()
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid["fwd_return_1w"] = grid["SPY_close"].shift(-FWD_HOURS) / grid["SPY_close"] - 1.0
    grid["raw_regime"] = grid["fwd_return_1w"].apply(label_regime)
    grid["label_available_ts"] = grid["timestamp"].shift(-FWD_HOURS)
    grid["feature_ok"] = grid[FEATURES + ["SPY_close", "QQQ_close"]].notna().all(axis=1)
    grid["row_pos"] = np.arange(len(grid))
    return grid


def working_frame(grid: pd.DataFrame) -> pd.DataFrame:
    need = ["timestamp", "SPY_close", "QQQ_close", "raw_regime", "label_available_ts", "feature_ok", *FEATURES]
    out = grid[need].copy()
    out = out[(out["timestamp"] >= TRAIN_START) & out["feature_ok"] & out["raw_regime"].notna()].copy()
    out["raw_regime"] = out["raw_regime"].astype(int)
    out = out.reset_index(drop=True)
    out["work_pos"] = np.arange(len(out))
    return out


def reference_walkforward() -> pd.DataFrame:
    ref = read_csv(REFERENCE_WALKFORWARD_PATH)
    ref = ref[(ref["timestamp"] >= EVAL_START) & (ref["timestamp"] <= EVAL_END)].copy()
    ref["predicted_regime"] = pd.to_numeric(ref["predicted_regime"], errors="coerce").astype("Int64")
    return ref.dropna(subset=["predicted_regime"]).reset_index(drop=True)


def canonical_eval_frame(work: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    ref_ts = set(ref["timestamp"])
    out = work[
        (work["timestamp"] >= EVAL_START)
        & (work["timestamp"] <= EVAL_END)
        & work["timestamp"].isin(ref_ts)
    ].copy()
    return out.sort_values("timestamp").reset_index(drop=True)


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


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[TemporalAttentionClassifier, dict[str, Any]]:
    seed_everything(seed)
    model = TemporalAttentionClassifier(input_size=x_train.shape[-1], hidden_size=32, num_layers=1, dropout=0.2).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "epochs_run": len(history), "history": history}


def predict_model(model: TemporalAttentionClassifier, x_test: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(int)


@dataclass
class WindowResult:
    variant: str
    window: int
    signals: pd.DataFrame
    fold_manifest: pd.DataFrame


def run_attention_window(work: pd.DataFrame, canonical: pd.DataFrame, window: int, device: torch.device) -> WindowResult:
    variant = f"TFT_attention_raw_{window}h"
    rng = np.random.default_rng(RANDOM_SEED + window)
    all_features = work[FEATURES].to_numpy(dtype=np.float32)
    timestamps = work["timestamp"].reset_index(drop=True)
    raw_labels = work["raw_regime"].to_numpy(dtype=int)
    label_available = pd.to_datetime(work["label_available_ts"]).reset_index(drop=True)
    canonical_pos = dict(zip(work["timestamp"], work["work_pos"]))
    signals, manifests = [], []

    for fold_num, (fold_start, fold_end) in enumerate(monthly_folds(), start=1):
        fit_mask = (timestamps < fold_start).to_numpy()
        scaler = StandardScaler()
        scaler.fit(all_features[fit_mask])
        scaled = scaler.transform(all_features).astype(np.float32)

        train_end_mask = (
            (timestamps >= TRAIN_START)
            & (timestamps < fold_start)
            & (label_available < fold_start)
            & (work["work_pos"] >= window - 1)
        ).to_numpy()
        train_end_positions = np.flatnonzero(train_end_mask)
        if len(train_end_positions) < 1000:
            continue
        split_idx = int(len(train_end_positions) * 0.8)
        train_pos = train_end_positions[:split_idx]
        val_pos = train_end_positions[split_idx:]
        if len(val_pos) < 100 or len(train_pos) < 1000:
            val_count = max(100, int(len(train_end_positions) * 0.15))
            train_pos = train_end_positions[:-val_count]
            val_pos = train_end_positions[-val_count:]

        y_train_full = raw_labels[train_pos]
        sample_idx = balanced_sample_indices(y_train_full, rng)
        train_pos_sample = train_pos[sample_idx]
        y_train = raw_labels[train_pos_sample]
        y_val = raw_labels[val_pos]
        x_train = sequence_tensor(scaled, train_pos_sample, window)
        x_val = sequence_tensor(scaled, val_pos, window)

        model, train_info = train_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            device=device,
            seed=RANDOM_SEED + window * 100 + fold_num,
        )

        test_df = canonical[(canonical["timestamp"] >= fold_start) & (canonical["timestamp"] < fold_end)].copy()
        test_positions = np.array([canonical_pos[t] for t in test_df["timestamp"] if t in canonical_pos], dtype=int)
        test_positions = test_positions[test_positions >= window - 1]
        test_df = test_df[test_df["timestamp"].isin(timestamps.iloc[test_positions].to_list())].copy()
        if test_df.empty:
            continue
        x_test = sequence_tensor(scaled, test_positions, window)
        test_df["predicted_regime"] = predict_model(model, x_test, device)
        fold_signal = test_df[["timestamp", "raw_regime", "predicted_regime"]].copy()
        fold_signal["variant"] = variant
        fold_signal["sequence_window_hours"] = window
        fold_signal["regime_name"] = fold_signal["predicted_regime"].map(REGIME_NAME)
        fold_signal["allocation_scale"] = 1.0
        fold_signal = fold_signal[
            ["timestamp", "variant", "sequence_window_hours", "raw_regime", "predicted_regime", "regime_name", "allocation_scale"]
        ]
        signals.append(fold_signal)
        manifests.append(
            {
                "variant": variant,
                "sequence_window_hours": window,
                "fold_start": fold_start,
                "fold_end": fold_end,
                "scaler_fit_start": timestamps[fit_mask].min(),
                "scaler_fit_end": timestamps[fit_mask].max(),
                "label_available_rule": "train sequence label timestamp + 168-row horizon must be strictly before fold_start",
                "train_sequence_count_before_sampling": len(train_pos),
                "train_sequence_count_after_balancing": len(train_pos_sample),
                "validation_sequence_count": len(val_pos),
                "test_sequence_count": len(test_df),
                "train_label_distribution_before_sampling": dist(raw_labels[train_pos]),
                "train_label_distribution_after_sampling": dist(y_train),
                "validation_label_distribution": dist(y_val),
                "prediction_distribution": dist(test_df["predicted_regime"]),
                "architecture": {
                    "type": "TFT-style temporal attention classifier",
                    "encoder": "input projection + single-layer GRU + attention pooling over time",
                    "layers": 1,
                    "hidden_size": 32,
                    "dropout": 0.2,
                    "batch_size": BATCH_SIZE,
                    "max_epochs": MAX_EPOCHS,
                    "patience": PATIENCE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "class_weighting": True,
                    "balanced_sampling_max_per_class": MAX_TRAIN_SAMPLES_PER_CLASS,
                },
                "early_stopping": "chronological validation split inside train period only; test fold never used",
                "best_val_loss": train_info["best_val_loss"],
                "epochs_run": train_info["epochs_run"],
                "strict_pit": True,
                "feature_list": FEATURES,
                "train_input_hash": frame_hash(
                    work.iloc[train_pos_sample][["timestamp", "raw_regime", "label_available_ts", *FEATURES]]
                ),
                "output_signal_hash": frame_hash(fold_signal),
            }
        )
        del model, x_train, x_val, x_test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return WindowResult(variant, window, pd.concat(signals, ignore_index=True), pd.DataFrame(manifests))


def price_update_frame(grid: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    sig = signals[["timestamp", "predicted_regime", "allocation_scale"]].drop_duplicates("timestamp").copy()
    data = grid[["timestamp", "SPY_close", "QQQ_close"]].merge(sig, on="timestamp", how="inner").sort_values("timestamp")
    data = data[(data["timestamp"] >= EVAL_START) & (data["timestamp"] <= EVAL_END)].copy()
    data[["SPY_close", "QQQ_close"]] = data[["SPY_close", "QQQ_close"]].ffill()
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
        "start_timestamp": daily["timestamp"].iloc[0],
        "end_timestamp": daily["timestamp"].iloc[-1],
        "start_equity": INITIAL_CAPITAL,
        "final_equity": final_equity,
        "CAGR": (final_equity / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0,
        "Sharpe": ret.mean() / ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0,
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0)
        if len(downside) > 1 and downside.std(ddof=1)
        else 0.0,
        "Max DD": float(dd.min()),
    }


def run_backtest(grid: pd.DataFrame, signals: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = price_update_frame(grid, signals)
    if len(data) < 2:
        raise ValueError(f"Not enough price-update rows for {variant}")
    equity = [INITIAL_CAPITAL]
    spy_equity = [INITIAL_CAPITAL]
    applied_regimes = [np.nan]
    target_assets = ["START"]
    target_scales = [0.0]
    gross_returns = [0.0]
    strategy_returns = [0.0]
    switch_costs = [0.0]
    position_changed = [False]
    active_asset = "START"
    active_scale = 0.0
    for idx in range(1, len(data)):
        signal_idx = idx - 1
        regime = int(data["predicted_regime"].iloc[signal_idx])
        scale = float(data["allocation_scale"].iloc[signal_idx])
        target_asset = REGIME_TO_ASSET.get(regime, "CASH")
        spy_ret = data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0
        qqq_ret = data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0
        if target_asset == "QQQ":
            gross_ret = scale * qqq_ret
        elif target_asset == "SPY":
            gross_ret = scale * spy_ret
        else:
            gross_ret = 0.0
            scale = 0.0
        changed = target_asset != active_asset or abs(scale - active_scale) > 1e-12
        cost = SWITCH_COST if changed else 0.0
        strat_ret = gross_ret - cost
        equity.append(equity[-1] * (1.0 + strat_ret))
        spy_equity.append(spy_equity[-1] * (1.0 + spy_ret))
        applied_regimes.append(regime)
        target_assets.append(target_asset)
        target_scales.append(scale)
        gross_returns.append(gross_ret)
        strategy_returns.append(strat_ret)
        switch_costs.append(cost)
        position_changed.append(changed)
        active_asset = target_asset
        active_scale = scale
    out = data.copy()
    out["variant"] = variant
    out["applied_regime"] = applied_regimes
    out["target_asset"] = target_assets
    out["target_scale"] = target_scales
    out["gross_strategy_return"] = gross_returns
    out["switch_cost"] = switch_costs
    out["strategy_return"] = strategy_returns
    out["position_changed"] = position_changed
    out["strategy_equity"] = equity
    out["spy_return"] = out["SPY_close"].pct_change().fillna(0.0)
    out["spy_equity"] = spy_equity
    out["drawdown"] = out["strategy_equity"] / out["strategy_equity"].cummax() - 1.0
    out["report_date"] = pd.to_datetime(out["timestamp"]).dt.date.astype(str)
    daily = out.loc[out.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    daily["strategy_daily_return"] = daily["strategy_equity"].pct_change().fillna(0.0)
    daily["spy_daily_return"] = daily["spy_equity"].pct_change().fillna(0.0)
    metrics = metrics_from_daily(daily, variant)
    metrics["hourly_price_update_rows"] = len(out)
    metrics["daily_rows"] = len(daily)
    metrics["switch_count"] = int(out["position_changed"].astype(bool).sum())
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


def monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    month_key = pd.to_datetime(daily["timestamp"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in daily.groupby(month_key):
        rows.append(
            {
                "month": month,
                "strategy_return": float(group["strategy_equity"].iloc[-1] / group["strategy_equity"].iloc[0] - 1.0),
                "start_equity": float(group["strategy_equity"].iloc[0]),
                "end_equity": float(group["strategy_equity"].iloc[-1]),
                "rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def signal_match(candidate: pd.DataFrame, reference: pd.DataFrame, name: str) -> dict[str, Any]:
    cand = candidate[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "candidate"})
    ref = reference[["timestamp", "predicted_regime"]].rename(columns={"predicted_regime": "reference"})
    cmp = cand.merge(ref, on="timestamp", how="outer", indicator=True)
    both = cmp[cmp["_merge"] == "both"].copy()
    if not both.empty:
        both["candidate"] = pd.to_numeric(both["candidate"], errors="coerce").astype("Int64")
        both["reference"] = pd.to_numeric(both["reference"], errors="coerce").astype("Int64")
        both["match"] = both["candidate"] == both["reference"]
    return {
        "comparison": name,
        "candidate_rows": len(cand),
        "reference_rows": len(ref),
        "overlap_rows": len(both),
        "candidate_only_rows": int((cmp["_merge"] == "left_only").sum()),
        "reference_only_rows": int((cmp["_merge"] == "right_only").sum()),
        "match_rate": float(both["match"].mean()) if len(both) else np.nan,
        "mismatch_count": int((both["match"] != True).sum()) if len(both) else None,
    }


def regime_distribution(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in signals.groupby("variant"):
        vc = group["predicted_regime"].astype(int).value_counts().sort_index()
        total = int(vc.sum())
        for regime in [0, 1, 2]:
            count = int(vc.get(regime, 0))
            rows.append(
                {
                    "variant": variant,
                    "regime": regime,
                    "regime_name": REGIME_NAME[regime],
                    "count": count,
                    "share": count / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def load_baseline_signals() -> dict[str, pd.DataFrame]:
    out = {}
    for name, path in BASELINE_SIGNAL_FILES.items():
        df = read_csv(path)
        df["predicted_regime"] = pd.to_numeric(df["predicted_regime"], errors="coerce").astype("Int64")
        out[name] = df.dropna(subset=["predicted_regime"]).copy()
        out[name]["predicted_regime"] = out[name]["predicted_regime"].astype(int)
    return out


def write_readme(summary: pd.DataFrame, best: dict[str, Any], conclusion: dict[str, Any]) -> None:
    lines = [
        "# MUB-27 TFT-Style Strict-PIT Research",
        "",
        "This is isolated research output only.",
        "",
        "## Governance",
        "",
        "- No production V4 Delta tables, Saxo execution, token refresh, RT/dashboard notebooks, live schedules, model registry assets, or trading logic were modified.",
        "- TFT-style attention training uses raw forward-return regime labels only. Full-sample teacher labels are not used for training.",
        "- For each monthly fold, scaler fit, sequence training labels, validation split, and early stopping are all limited to train-period information.",
        "- Train labels are additionally restricted so the 168-row forward-return label is realized strictly before fold start.",
        "",
        "## Comparison",
        "",
        md_table(summary),
        "",
        "## Best TFT-Style Attention",
        "",
        json.dumps(jsonable(best), indent=2, sort_keys=True),
        "",
        "## Conclusion",
        "",
        json.dumps(jsonable(conclusion), indent=2, sort_keys=True),
        "",
    ]
    write_text("\n".join(lines), OUTPUT_ROOT / "README.md")


def main() -> None:
    seed_everything()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = load_grid()
    work = working_frame(grid)
    ref = reference_walkforward()
    canonical = canonical_eval_frame(work, ref)
    baselines = load_baseline_signals()

    window_results = [run_attention_window(work, canonical, window, device) for window in SEQUENCE_WINDOWS]
    tft_signals = pd.concat([result.signals for result in window_results], ignore_index=True)
    write_csv(tft_signals, SIGNALS_DIR / "tft_signals.csv")
    fold_manifest = pd.concat([result.fold_manifest for result in window_results], ignore_index=True)
    write_csv(fold_manifest, MANIFESTS_DIR / "fold_manifest.csv")
    write_json({"folds": fold_manifest.to_dict("records")}, MANIFESTS_DIR / "fold_manifest.json")

    summary_rows = []
    match_rows = []
    combined_daily_returns = []
    combined_equity_daily = []
    combined_yearly_returns = []
    for result in window_results:
        hourly, daily, metrics = run_backtest(grid, result.signals, result.variant)
        write_csv(hourly, PERFORMANCE_DIR / f"{result.variant}_equity_curve_hourly_price_updates.csv")
        write_csv(daily, PERFORMANCE_DIR / f"{result.variant}_equity_curve_daily.csv")
        daily_returns = daily[
            ["timestamp", "variant", "report_date", "strategy_daily_return", "spy_daily_return", "strategy_equity", "spy_equity"]
        ].copy()
        annual = yearly_returns(daily)
        annual.insert(0, "variant", result.variant)
        write_csv(daily_returns, PERFORMANCE_DIR / f"{result.variant}_daily_returns.csv")
        write_csv(annual, PERFORMANCE_DIR / f"{result.variant}_yearly_returns.csv")
        write_csv(monthly_returns(daily), PERFORMANCE_DIR / f"{result.variant}_monthly_returns.csv")
        combined_daily_returns.append(daily_returns)
        combined_equity_daily.append(daily)
        combined_yearly_returns.append(annual)
        match_b = signal_match(result.signals, baselines["B_reference_teacher_smoothed"], f"{result.variant}_vs_B_C2")
        bilstm_24 = read_csv(BILSTM_SIGNAL_PATH)
        bilstm_24 = bilstm_24[bilstm_24["variant"].eq("BiLSTM_raw_24h")].copy()
        bilstm_24["predicted_regime"] = pd.to_numeric(bilstm_24["predicted_regime"], errors="coerce").astype("Int64")
        bilstm_24 = bilstm_24.dropna(subset=["predicted_regime"])
        bilstm_24["predicted_regime"] = bilstm_24["predicted_regime"].astype(int)
        match_a = signal_match(result.signals, baselines["A_raw_label_walkforward"], f"{result.variant}_vs_A_raw_label_WF")
        match_lstm = signal_match(result.signals, bilstm_24, f"{result.variant}_vs_BiLSTM_24h")
        match_rows.extend([match_b, match_a, match_lstm])
        summary_rows.append(
            {
                "variant": result.variant,
                "model_family": "TFT_attention",
                "pit_status": "strict_pit",
                "sequence_window_hours": result.window,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "Sortino": metrics["Sortino"],
                "Max DD": metrics["Max DD"],
                "Final Equity": metrics["final_equity"],
                "Switch Count": metrics["switch_count"],
                "Match vs B/C2": match_b["match_rate"],
                "Match vs raw-label WF": match_a["match_rate"],
                "Match vs BiLSTM 24h": match_lstm["match_rate"],
            }
        )
    write_csv(pd.concat(combined_daily_returns, ignore_index=True), PERFORMANCE_DIR / "tft_daily_returns.csv")
    write_csv(pd.concat(combined_equity_daily, ignore_index=True), PERFORMANCE_DIR / "tft_equity_curve_daily.csv")
    write_csv(pd.concat(combined_yearly_returns, ignore_index=True), PERFORMANCE_DIR / "tft_yearly_returns.csv")

    prior = pd.read_csv(MUB27_COMPARISON_PATH)
    prior = prior.rename(columns={"Signal Match vs B/C2": "Match vs B/C2"})
    prior["model_family"] = "baseline"
    prior["sequence_window_hours"] = np.nan
    prior["Match vs raw-label WF"] = np.nan
    prior["Match vs BiLSTM 24h"] = np.nan
    prior = prior[
        [
            "variant",
            "model_family",
            "pit_status",
            "sequence_window_hours",
            "CAGR",
            "Sharpe",
            "Sortino",
            "Max DD",
            "Final Equity",
            "Switch Count",
            "Match vs B/C2",
            "Match vs raw-label WF",
            "Match vs BiLSTM 24h",
        ]
    ]
    bilstm_comparison = pd.read_csv(BILSTM_COMPARISON_PATH)
    bilstm_comparison = bilstm_comparison[bilstm_comparison["model_family"].eq("BiLSTM")].copy()
    bilstm_comparison["Match vs BiLSTM 24h"] = np.where(
        bilstm_comparison["variant"].eq("BiLSTM_raw_24h"), 1.0, np.nan
    )
    bilstm_comparison = bilstm_comparison[
        [
            "variant",
            "model_family",
            "pit_status",
            "sequence_window_hours",
            "CAGR",
            "Sharpe",
            "Sortino",
            "Max DD",
            "Final Equity",
            "Switch Count",
            "Match vs B/C2",
            "Match vs raw-label WF",
            "Match vs BiLSTM 24h",
        ]
    ]
    tft_summary = pd.DataFrame(summary_rows)
    comparison = pd.concat([prior, bilstm_comparison, tft_summary], ignore_index=True)
    write_csv(comparison, FORENSICS_DIR / "comparison_vs_A_B_C_D_SPY_BiLSTM.csv")
    write_csv(pd.DataFrame(match_rows), FORENSICS_DIR / "match_rates.csv")
    write_csv(regime_distribution(tft_signals), FORENSICS_DIR / "regime_distribution.csv")

    best = tft_summary.sort_values(["CAGR", "Sharpe"], ascending=False).iloc[0].to_dict()
    a = prior[prior["variant"].eq("A_raw_label_walkforward")].iloc[0].to_dict()
    b = prior[prior["variant"].eq("B_reference_teacher_smoothed")].iloc[0].to_dict()
    bilstm_best = bilstm_comparison[bilstm_comparison["variant"].eq("BiLSTM_raw_24h")].iloc[0].to_dict()
    spy = prior[prior["variant"].eq("E_spy_benchmark")].iloc[0].to_dict()
    best["gap_vs_raw_label_WF_CAGR"] = float(best["CAGR"] - a["CAGR"])
    best["gap_vs_SPY_CAGR"] = float(best["CAGR"] - spy["CAGR"])
    best["gap_vs_BiLSTM_24h_CAGR"] = float(best["CAGR"] - bilstm_best["CAGR"])
    best["gap_vs_B_C2_CAGR"] = float(best["CAGR"] - b["CAGR"])
    best["gap_vs_B_C2_final_equity"] = float(best["Final Equity"] - b["Final Equity"])
    conclusion = {
        "tft_beats_raw_label_WF": bool(best["CAGR"] > a["CAGR"]),
        "tft_beats_BiLSTM_24h": bool(best["CAGR"] > bilstm_best["CAGR"]),
        "tft_beats_SPY": bool(best["CAGR"] > spy["CAGR"]),
        "best_tft_variant": best["variant"],
        "best_tft_gap_vs_B_C2_CAGR": best["gap_vs_B_C2_CAGR"],
        "recommendation": "move_to_label_redesign" if best["CAGR"] <= max(a["CAGR"], bilstm_best["CAGR"], spy["CAGR"]) else "continue_TFT",
        "note": "This first TFT-style attention leg uses raw labels only; no full-sample teacher labels are used for training.",
    }
    write_json({"best_tft": best, "conclusion": conclusion}, FORENSICS_DIR / "tft_conclusion.json")

    input_hashes = {
        "faithful_legacy_grid": {
            "path": FAITHFUL_GRID_PATH,
            "sha256": file_sha256(FAITHFUL_GRID_PATH),
            "rows": len(grid),
            "min_timestamp": grid["timestamp"].min(),
            "max_timestamp": grid["timestamp"].max(),
        },
        "reference_walkforward_signals": {
            "path": REFERENCE_WALKFORWARD_PATH,
            "sha256": file_sha256(REFERENCE_WALKFORWARD_PATH),
            "rows": len(ref),
        },
        "reference_full_signal_export": {
            "path": REFERENCE_FULL_SIGNAL_PATH,
            "sha256": file_sha256(REFERENCE_FULL_SIGNAL_PATH),
        },
        "mub27_strict_pit_comparison": {
            "path": MUB27_COMPARISON_PATH,
            "sha256": file_sha256(MUB27_COMPARISON_PATH),
        },
        "bilstm_comparison": {
            "path": BILSTM_COMPARISON_PATH,
            "sha256": file_sha256(BILSTM_COMPARISON_PATH),
        },
        "bilstm_signals": {
            "path": BILSTM_SIGNAL_PATH,
            "sha256": file_sha256(BILSTM_SIGNAL_PATH),
        },
        "baseline_signal_files": {
            name: {"path": path, "sha256": file_sha256(path)} for name, path in BASELINE_SIGNAL_FILES.items()
        },
        "canonical_eval": {
            "rows": len(canonical),
            "min_timestamp": canonical["timestamp"].min(),
            "max_timestamp": canonical["timestamp"].max(),
            "timestamp_hash": frame_hash(canonical[["timestamp"]]),
        },
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
        "device": str(device),
        "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        "output_root": OUTPUT_ROOT,
        "features": FEATURES,
        "sequence_windows": SEQUENCE_WINDOWS,
        "raw_label_definition": {
            "forward_rows": FWD_HOURS,
            "expansion": "SPY forward return > +0.5%",
            "slowdown": "-0.5% <= SPY forward return <= +0.5%",
            "recession": "SPY forward return < -0.5%",
        },
        "strict_pit_controls": [
            "scaler fit only on rows before fold_start",
            "sequence inputs contain only observations up to sequence end timestamp",
            "train labels require label_available_ts strictly before fold_start",
            "early stopping uses chronological train-period validation only",
            "test fold is never used for scaling, training, label smoothing, or early stopping",
        ],
        "performance_methodology": {
            "allocation": "prior-bar / next-bar",
            "row_filter": "price-update hourly rows only",
            "switch_cost_bps": 5,
            "daily_reporting": "daily-last equity",
            "cagr_period": "2021-01-01 through 2025-12-31, exactly 5.0 years",
            "sharpe_sortino": "daily returns, 252 annualization, risk-free 0",
        },
        "production_delta_writes": False,
        "model_registry_writes": False,
        "saxo_or_rt_writes": False,
        "dashboard_writes": False,
        "workflow_or_schedule_changes": False,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")
    write_readme(comparison, best, conclusion)

    print(comparison.to_string(index=False))
    print(json.dumps(jsonable({"best_tft": best, "conclusion": conclusion}), indent=2, sort_keys=True))
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
