# Databricks notebook source
"""MUB-28 teacher-student April-cutoff retrain candidate.

Research / validation only. This notebook creates a versioned candidate model
and shadow validation table. It does not overwrite the live alias, model
registry, RT notebooks, Saxo execution, token refresher, dashboard, or jobs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pyspark.sql import functions as F


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
NS = f"{CATALOG}.{SCHEMA}"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")

MODEL_VERSION = "bt_teacher_student_smoothed_v4_2026_04_cutoff_2026_05_08"
VERSIONED_MODEL_PATH = BASE / "models" / f"{MODEL_VERSION}.joblib"
LIVE_ALIAS_PATH = BASE / "models" / "bt_fred_1d_xgboost_regime_live.joblib"
CURRENT_VERSIONED_PATH = BASE / "models" / "bt_teacher_student_smoothed_v4_2026_05_08.joblib"
CURRENT_SOURCE_FOLD_PATH = (
    BASE
    / "v4_1_csv"
    / "faithful_legacy_grid_model_replay_2026-04-30"
    / "models"
    / "folds"
    / "2025_12"
    / "model.joblib"
)

OUTPUT_ROOT = BASE / "mub28_teacher_student_april_cutoff_2026-05-08"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
VALIDATION_DIR = OUTPUT_ROOT / "validation"
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
SIGNALS_DIR = OUTPUT_ROOT / "signals"

SILVER_BT_TABLE = f"{NS}.silver_bt_master_grid_1h"
SILVER_RT_TABLE = f"{NS}.silver_rt_1h"
BRONZE_BT_MARKET_TABLE = f"{NS}.bronze_bt_market_1h"
BRONZE_RT_MARKET_TABLE = f"{NS}.bronze_rt_market_1h"
REGISTRY_TABLE = f"{NS}.model_registry_current"
SHADOW_TABLE = f"{NS}.bronze_rt_regime_signal_teacher_student_april_shadow"

TRAIN_START = pd.Timestamp("2016-01-01")
REQUESTED_CUTOFF = pd.Timestamp("2026-04-30 23:59:59")
FWD_HOURS = 168
INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005

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
FEATURE_SCHEMA_SHA256 = hashlib.sha256(json.dumps(FEATURES).encode("utf-8")).hexdigest()

REGIME_NAME = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
POSITION_MAP = {0: "100% QQQ", 1: "100% SPY", 2: "100% CASH"}
REGIME_TO_ASSET = {0: "QQQ", 1: "SPY", 2: "CASH"}

RECOVERED_TEACHER_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.03,
    "objective": "multi:softmax",
    "num_class": 3,
    "tree_method": "hist",
    "random_state": 42,
}
RECOVERED_STUDENT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "multi:softmax",
    "num_class": 3,
    "tree_method": "hist",
    "random_state": 42,
}


def ensure_dirs() -> None:
    for path in [MANIFESTS_DIR, VALIDATION_DIR, PERFORMANCE_DIR, SIGNALS_DIR, VERSIONED_MODEL_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.sort_index(axis=1).to_csv(index=False).encode("utf-8")).hexdigest()


def normalize_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    try:
        out[column] = out[column].dt.tz_localize(None)
    except Exception:
        pass
    return out.dropna(subset=[column]).sort_values(column).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside MUB-28 output root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
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

    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside MUB-28 output root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def label_regime(ret: float) -> float:
    if pd.isna(ret):
        return np.nan
    if ret > 0.005:
        return 0
    if ret < -0.005:
        return 2
    return 1


def dist(series: pd.Series) -> dict[str, int]:
    vc = pd.Series(series).dropna().astype(int).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


def value_dist(series: pd.Series) -> dict[str, int]:
    vc = pd.Series(series).dropna().astype(str).value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}


def load_feature_grid() -> pd.DataFrame:
    grid = spark.table(SILVER_BT_TABLE).select(["timestamp", "SPY_close", "QQQ_close", *FEATURES]).toPandas()
    grid = normalize_timestamp(grid)
    for col in ["SPY_close", "QQQ_close", *FEATURES]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    return grid


def load_label_prices(start_ts: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    bt = (
        spark.table(BRONZE_BT_MARKET_TABLE)
        .filter(F.col("symbol").isin("SPY", "QQQ"))
        .select("symbol", "timestamp", "close")
        .toPandas()
    )
    rt = (
        spark.table(BRONZE_RT_MARKET_TABLE)
        .filter(F.col("symbol").isin("SPY", "QQQ"))
        .select("symbol", "timestamp", "close")
        .toPandas()
    )
    bt = normalize_timestamp(bt)
    rt = normalize_timestamp(rt)
    bt["priority"] = 0
    rt["priority"] = 1
    market = (
        pd.concat([bt, rt], ignore_index=True)
        .sort_values(["symbol", "timestamp", "priority"])
        .drop_duplicates(["symbol", "timestamp"], keep="last")
    )
    wide = (
        market.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .reset_index()
        .sort_values("timestamp")
    )
    latest_by_symbol = market.groupby("symbol")["timestamp"].max().astype(str).to_dict()
    hours = pd.DataFrame({"label_ts": pd.date_range(start=start_ts, end=wide["timestamp"].max(), freq="h")})
    price = hours.merge(wide.rename(columns={"timestamp": "label_ts"}), on="label_ts", how="left").sort_values("label_ts")
    price[["SPY", "QQQ"]] = price[["SPY", "QQQ"]].ffill()
    price = price.rename(columns={"SPY": "label_SPY_close", "QQQ": "label_QQQ_close"})
    return price, {
        "market_union_max_ts_by_symbol": latest_by_symbol,
        "hourly_label_grid_max_ts": price["label_ts"].max(),
        "label_price_rows": len(price),
    }


def attach_labels(grid: pd.DataFrame, label_prices: pd.DataFrame) -> pd.DataFrame:
    work = grid.copy()
    work["label_ts"] = work["timestamp"] + pd.Timedelta(hours=FWD_HOURS)
    work = work.merge(label_prices, on="label_ts", how="left")
    work["fwd_return_1w"] = work["label_SPY_close"] / work["SPY_close"] - 1.0
    work["raw_regime"] = work["fwd_return_1w"].apply(label_regime)
    work["label_safe"] = (
        work["timestamp"].ge(TRAIN_START)
        & work["timestamp"].le(REQUESTED_CUTOFF)
        & work[FEATURES].notna().all(axis=1)
        & work["SPY_close"].notna()
        & work["QQQ_close"].notna()
        & work["label_SPY_close"].notna()
        & work["raw_regime"].notna()
    )
    return work


def assert_schema_compatible(model_obj: Any, name: str) -> None:
    if hasattr(model_obj, "feature_names_in_"):
        got = list(model_obj.feature_names_in_)
        if got != FEATURES:
            raise ValueError(f"{name} feature_names_in_ mismatch: {got}")
    if hasattr(model_obj, "n_features_in_") and int(model_obj.n_features_in_) != len(FEATURES):
        raise ValueError(f"{name} n_features_in_ mismatch: {model_obj.n_features_in_}")


def train_candidate(trainable: pd.DataFrame, cutoff_ts: pd.Timestamp) -> tuple[Any, Any, pd.DataFrame]:
    trainable = trainable[(trainable["timestamp"] >= TRAIN_START) & (trainable["timestamp"] <= cutoff_ts)].copy()
    teacher = xgb.XGBClassifier(**RECOVERED_TEACHER_PARAMS)
    teacher.fit(trainable[FEATURES], trainable["raw_regime"].astype(int))
    pseudo = trainable[["timestamp", "raw_regime", *FEATURES]].copy()
    pseudo["teacher_pseudo_regime"] = teacher.predict(pseudo[FEATURES].fillna(0)).astype(int)

    student = xgb.XGBClassifier(**RECOVERED_STUDENT_PARAMS)
    student.fit(pseudo[FEATURES], pseudo["teacher_pseudo_regime"].astype(int))
    assert_schema_compatible(student, "candidate_student")
    joblib.dump(student, VERSIONED_MODEL_PATH)
    return teacher, student, pseudo


def prediction_frame(model_obj: Any, features_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = features_df[["timestamp", *FEATURES]].copy()
    preds = model_obj.predict(out[FEATURES].fillna(0)).astype(int)
    out[f"{prefix}_regime"] = preds
    out[f"{prefix}_regime_name"] = pd.Series(preds).map(REGIME_NAME).to_numpy()
    out[f"{prefix}_position"] = pd.Series(preds).map(POSITION_MAP).to_numpy()
    return out[["timestamp", f"{prefix}_regime", f"{prefix}_regime_name", f"{prefix}_position"]]


def validate_latest_rt(current_model: Any, candidate_model: Any) -> dict[str, Any]:
    rt = spark.table(SILVER_RT_TABLE).orderBy(F.col("timestamp").desc()).limit(100).toPandas()
    rt = normalize_timestamp(rt).sort_values("timestamp").reset_index(drop=True)
    for col in FEATURES:
        rt[col] = pd.to_numeric(rt[col], errors="coerce")
    null_counts = {col: int(rt[col].isna().sum()) for col in FEATURES}
    current = prediction_frame(current_model, rt, "current")
    candidate = prediction_frame(candidate_model, rt, "candidate")
    cmp = current.merge(candidate, on="timestamp").merge(rt[["timestamp", *FEATURES]], on="timestamp")
    cmp["candidate_model_version"] = MODEL_VERSION
    cmp["candidate_model_path"] = str(VERSIONED_MODEL_PATH)
    cmp["validation_source"] = "latest_100_silver_rt_1h"
    cmp["created_ts"] = datetime.now(timezone.utc).replace(tzinfo=None)
    write_csv(cmp, VALIDATION_DIR / "latest_100_silver_rt_candidate_vs_current.csv")

    spark.createDataFrame(cmp).write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(SHADOW_TABLE)
    latest = cmp.sort_values("timestamp").iloc[-1].to_dict()
    return {
        "validation_table_path": str(VALIDATION_DIR / "latest_100_silver_rt_candidate_vs_current.csv"),
        "shadow_table": SHADOW_TABLE,
        "rows": len(cmp),
        "feature_null_counts": null_counts,
        "candidate_regime_distribution": dist(cmp["candidate_regime"]),
        "current_regime_distribution": dist(cmp["current_regime"]),
        "candidate_position_distribution": value_dist(cmp["candidate_position"]),
        "current_position_distribution": value_dist(cmp["current_position"]),
        "latest_row": latest,
        "regime_diff_rows": int((cmp["candidate_regime"] != cmp["current_regime"]).sum()),
    }


def signal_from_model(model_obj: Any, grid: pd.DataFrame, name: str, start: str, end_ts: pd.Timestamp) -> pd.DataFrame:
    data = grid[(grid["timestamp"] >= pd.Timestamp(start)) & (grid["timestamp"] <= end_ts)].copy()
    data = data.dropna(subset=["SPY_close", "QQQ_close"]).sort_values("timestamp")
    data["predicted_regime"] = model_obj.predict(data[FEATURES].fillna(0)).astype(int)
    data["variant"] = name
    data["allocation_scale"] = 1.0
    data["regime_name"] = data["predicted_regime"].map(REGIME_NAME)
    return data[["timestamp", "variant", "predicted_regime", "regime_name", "allocation_scale"]]


def spy_signal(grid: pd.DataFrame, start: str, end_ts: pd.Timestamp) -> pd.DataFrame:
    data = grid[(grid["timestamp"] >= pd.Timestamp(start)) & (grid["timestamp"] <= end_ts)].copy()
    data["variant"] = "SPY_benchmark"
    data["predicted_regime"] = 1
    data["regime_name"] = "Slowdown"
    data["allocation_scale"] = 1.0
    return data[["timestamp", "variant", "predicted_regime", "regime_name", "allocation_scale"]]


def backtest(grid: pd.DataFrame, signals: pd.DataFrame, variant: str) -> dict[str, Any]:
    sig = signals[["timestamp", "predicted_regime", "allocation_scale"]].drop_duplicates("timestamp")
    data = grid[["timestamp", "SPY_close", "QQQ_close"]].merge(sig, on="timestamp", how="inner").sort_values("timestamp")
    data[["SPY_close", "QQQ_close"]] = data[["SPY_close", "QQQ_close"]].ffill()
    changed_price = data[["SPY_close", "QQQ_close"]].ne(data[["SPY_close", "QQQ_close"]].shift()).any(axis=1)
    data = data[changed_price].dropna(subset=["SPY_close", "QQQ_close", "predicted_regime"]).reset_index(drop=True)
    if len(data) < 2:
        raise ValueError(f"Not enough rows to backtest {variant}")
    equity = [INITIAL_CAPITAL]
    active = "START"
    pos_changed = [False]
    for idx in range(1, len(data)):
        regime = int(data["predicted_regime"].iloc[idx - 1])
        asset = REGIME_TO_ASSET.get(regime, "CASH")
        if asset == "QQQ":
            gross = data["QQQ_close"].iloc[idx] / data["QQQ_close"].iloc[idx - 1] - 1.0
        elif asset == "SPY":
            gross = data["SPY_close"].iloc[idx] / data["SPY_close"].iloc[idx - 1] - 1.0
        else:
            gross = 0.0
        changed = asset != active
        ret = gross - (SWITCH_COST if changed else 0.0)
        equity.append(equity[-1] * (1.0 + ret))
        pos_changed.append(changed)
        active = asset
    data["strategy_equity"] = equity
    data["position_changed"] = pos_changed
    data["drawdown"] = data["strategy_equity"] / data["strategy_equity"].cummax() - 1.0
    data["report_date"] = data["timestamp"].dt.date.astype(str)
    daily = data.loc[data.groupby("report_date")["timestamp"].idxmax()].sort_values("timestamp")
    ret = daily["strategy_equity"].pct_change().dropna()
    downside = ret[ret < 0]
    years = max((daily["timestamp"].iloc[-1] - daily["timestamp"].iloc[0]).days / 365.25, 1e-9)
    final = float(daily["strategy_equity"].iloc[-1])
    return {
        "variant": variant,
        "start_timestamp": daily["timestamp"].iloc[0],
        "end_timestamp": daily["timestamp"].iloc[-1],
        "CAGR": (final / INITIAL_CAPITAL) ** (1.0 / years) - 1.0,
        "Sharpe": ret.mean() / ret.std(ddof=1) * math.sqrt(252.0) if ret.std(ddof=1) else 0.0,
        "Sortino": ret.mean() / downside.std(ddof=1) * math.sqrt(252.0) if len(downside) > 1 and downside.std(ddof=1) else 0.0,
        "Max DD": float(data["drawdown"].min()),
        "final_equity": final,
        "switch_count": int(pd.Series(pos_changed).astype(bool).sum()),
        "hourly_price_update_rows": len(data),
        "daily_rows": len(daily),
    }


def main() -> None:
    ensure_dirs()
    grid = load_feature_grid()
    label_prices, label_info = load_label_prices(grid["timestamp"].min())
    labeled = attach_labels(grid, label_prices)
    trainable = labeled[labeled["label_safe"]].copy()
    cutoff_ts = trainable["timestamp"].max()
    latest_label_ts = trainable.loc[trainable["timestamp"].idxmax(), "label_ts"]

    if cutoff_ts < pd.Timestamp("2026-04-30"):
        raise ValueError(f"April 2026 is not label-safe; latest safe cutoff is {cutoff_ts}")

    current_hash = file_sha256(CURRENT_VERSIONED_PATH)
    source_hash = file_sha256(CURRENT_SOURCE_FOLD_PATH)
    if current_hash != "7dd1949058c4af0e6525dff1daea095603e108cba2b0dc9fd5a4756968b44f6b":
        raise ValueError(f"Current artifact SHA mismatch: {current_hash}")
    if current_hash != source_hash:
        raise ValueError(f"Current promoted/source fold SHA mismatch: {current_hash} != {source_hash}")

    current_model = joblib.load(LIVE_ALIAS_PATH)
    source_model = joblib.load(CURRENT_SOURCE_FOLD_PATH)
    assert_schema_compatible(current_model, "current_live_alias")
    assert_schema_compatible(source_model, "source_2025_12_fold")

    teacher, candidate, pseudo = train_candidate(trainable, cutoff_ts)
    candidate_sha = file_sha256(VERSIONED_MODEL_PATH)
    write_csv(pseudo[["timestamp", "raw_regime", "teacher_pseudo_regime"]], SIGNALS_DIR / "candidate_teacher_pseudo_labels.csv")

    validation = validate_latest_rt(current_model, candidate)

    start = "2021-01-01"
    current_sig = signal_from_model(current_model, grid, "current_live_2025_12_static", start, cutoff_ts)
    cand_sig = signal_from_model(candidate, grid, "candidate_april_cutoff_static", start, cutoff_ts)
    spy_sig = spy_signal(grid, start, cutoff_ts)
    write_csv(current_sig, SIGNALS_DIR / "current_live_2025_12_static_signals.csv")
    write_csv(cand_sig, SIGNALS_DIR / "candidate_april_cutoff_static_signals.csv")
    write_csv(spy_sig, SIGNALS_DIR / "spy_benchmark_signals.csv")
    metrics = pd.DataFrame(
        [
            backtest(grid, current_sig, "current_live_2025_12_static"),
            backtest(grid, cand_sig, "candidate_april_cutoff_static"),
            backtest(grid, spy_sig, "SPY_benchmark"),
            {
                "variant": "prior_B_C2_benchmark_2021_2025",
                "CAGR": 0.208405,
                "Sharpe": 1.0024,
                "Sortino": 1.3729,
                "Max DD": -0.255385,
                "final_equity": 12_883_454.9938051,
                "switch_count": 296,
                "methodology_note": "Prior B/C2 recovered teacher-smoothed benchmark over 2021-2025; not extended into Apr 2026 in this run.",
            },
        ]
    )
    write_csv(metrics, PERFORMANCE_DIR / "candidate_vs_current_static_backtest_summary.csv")

    manifest = {
        "model_version": MODEL_VERSION,
        "candidate_model_path": VERSIONED_MODEL_PATH,
        "candidate_sha256": candidate_sha,
        "live_alias_unchanged": LIVE_ALIAS_PATH,
        "current_live_alias_sha256": file_sha256(LIVE_ALIAS_PATH),
        "current_versioned_model_path": CURRENT_VERSIONED_PATH,
        "current_source_fold_path": CURRENT_SOURCE_FOLD_PATH,
        "current_source_fold_name": "2025_12",
        "current_source_sha256": source_hash,
        "baseline_reproduction_status": "current versioned artifact is byte-identical to source 2025_12 fold and expected SHA",
        "train_start": TRAIN_START,
        "train_end": cutoff_ts,
        "latest_label_timestamp_used": latest_label_ts,
        "requested_cutoff": REQUESTED_CUTOFF,
        "april_2026_fully_label_safe": bool(cutoff_ts >= pd.Timestamp("2026-04-30 15:00:00")),
        "label_rule": "SPY close at timestamp + 168 hours, using hourly forward-filled union of bronze_bt_market_1h and bronze_rt_market_1h",
        "source_grid_table": SILVER_BT_TABLE,
        "source_market_tables": [BRONZE_BT_MARKET_TABLE, BRONZE_RT_MARKET_TABLE],
        "source_grid_path": SILVER_BT_TABLE,
        "feature_schema": FEATURES,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "train_rows": len(trainable),
        "teacher_params": RECOVERED_TEACHER_PARAMS,
        "student_params": RECOVERED_STUDENT_PARAMS,
        "teacher_label_distribution": dist(trainable["raw_regime"]),
        "student_pseudo_label_distribution": dist(pseudo["teacher_pseudo_regime"]),
        "validation": validation,
        "backtest_summary_path": PERFORMANCE_DIR / "candidate_vs_current_static_backtest_summary.csv",
        "runtime": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "xgboost_version": xgb.__version__,
            "joblib_version": joblib.__version__,
            "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
            "cluster_id": os.environ.get("DATABRICKS_CLUSTER_ID") or os.environ.get("DB_CLUSTER_ID"),
            "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_commit": git_value(["rev-parse", "HEAD"]),
        },
        "production_delta_writes": {"shadow_table_only": SHADOW_TABLE},
        "model_registry_writes": False,
        "live_alias_overwrite": False,
        "rt_notebook_changes": False,
        "saxo_execution_changes": False,
        "token_refresher_changes": False,
        "dashboard_changes": False,
        "workflow_or_schedule_changes": False,
    }
    manifest_path = MANIFESTS_DIR / f"{MODEL_VERSION}_manifest.json"
    write_json(manifest, manifest_path)
    print(json.dumps({"manifest_path": str(manifest_path), **manifest}, indent=2, default=str))


if __name__ == "__main__":
    main()
