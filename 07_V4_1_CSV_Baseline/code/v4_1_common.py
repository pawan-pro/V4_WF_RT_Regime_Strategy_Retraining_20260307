"""Shared helpers for the isolated V4.1 CSV-only baseline.

These helpers intentionally write only under V4_1_ROOT. They may read approved
historical source tables during the input-freeze step, but all downstream steps
consume CSV files only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
V4_1_ROOT = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data/v4_1_csv")
MARKET_SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_bt_market_1h"

PRODUCTION_TABLES = {
    f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h",
    f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals",
    f"{CATALOG}.{SCHEMA}.model_registry_current",
}

SUBDIRS = [
    "inputs",
    "grids",
    "models",
    "signals",
    "performance",
    "manifests",
    "code",
]

INDICATORS = [
    "CPIAUCSL",
    "PCEPI",
    "PCE",
    "PAYEMS",
    "UNRATE",
    "GDPC1",
    "INDPRO",
    "UMCSENT",
]

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

FINAL_GRID_COLUMNS = [
    "timestamp",
    "SPY_close",
    "QQQ_close",
    *FEATURES,
]

TIMING_VARIANTS = [
    "macro_market_open_or_30m_buffer__vix_next_day_10am",
    "macro_market_open_or_60m_buffer__vix_next_day_10am",
    "macro_plus_120m__vix_next_day_10am",
    "macro_next_grid_after_release__vix_next_day_10am",
]

REFERENCE_METRICS = {
    "cagr": 0.208405,
    "sharpe": 1.0024,
    "sortino": 1.3729,
    "max_drawdown": -0.255385,
    "final_equity": 12_883_454.99,
    "start_date": "2021-01-01",
    "end_date": "2025-12-31",
}

INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005
NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def resolve_root(root: str | os.PathLike[str] | None = None) -> Path:
    return Path(root) if root else V4_1_ROOT


def ensure_v4_1_dirs(root: str | os.PathLike[str] | None = None) -> Path:
    base = resolve_root(root)
    for subdir in SUBDIRS:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    return base


def assert_v4_1_path(path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> Path:
    base = resolve_root(root).resolve()
    candidate = Path(path).resolve()
    if base != candidate and base not in candidate.parents:
        raise ValueError(f"Refusing to write outside isolated V4.1 root: {candidate}")
    return candidate


def refuse_production_table_write(table_name: str) -> None:
    normalized = table_name.strip("`").lower()
    if normalized in {t.lower() for t in PRODUCTION_TABLES}:
        raise RuntimeError(f"Refusing to modify production table {table_name}")


def write_csv(df: pd.DataFrame, path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> None:
    out = assert_v4_1_path(path, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def write_json(payload: dict[str, Any], path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> None:
    out = assert_v4_1_path(path, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(path: str | os.PathLike[str], source_description: str, production_source_table: str | None = None) -> dict[str, Any]:
    p = Path(path)
    df = pd.read_csv(p, nrows=5)
    full = pd.read_csv(p)
    ts_cols = [c for c in ["timestamp", "release_ts_utc", "available_ts_utc", "release_date", "observation_date"] if c in full.columns]
    ranges: dict[str, dict[str, str | None]] = {}
    for col in ts_cols:
        s = pd.to_datetime(full[col], errors="coerce")
        ranges[col] = {
            "min": None if s.dropna().empty else str(s.min()),
            "max": None if s.dropna().empty else str(s.max()),
        }
    return {
        "file_path": str(p),
        "row_count": int(len(full)),
        "schema": list(df.columns),
        "date_ranges": ranges,
        "sha256": sha256_file(p),
        "source_description": source_description,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "production_source_table": production_source_table,
        "v4_1_consumption_note": "V4.1 consumes this frozen CSV, not the Delta source.",
    }


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def ny_market_open_for_date(date_like: Any) -> pd.Timestamp:
    d = pd.Timestamp(date_like).date()
    return pd.Timestamp(datetime.combine(d, time(9, 30)), tz=NY).tz_convert(UTC).tz_localize(None)


def ny_time_to_utc(date_like: Any, hhmm: str) -> pd.Timestamp:
    hour, minute = [int(x) for x in hhmm.split(":")]
    d = pd.Timestamp(date_like).date()
    return pd.Timestamp(datetime.combine(d, time(hour, minute)), tz=NY).tz_convert(UTC).tz_localize(None)


def ceil_hour(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    floored = ts.floor("h")
    return floored if ts == floored else floored + pd.Timedelta(hours=1)


def macro_available_ts(release_ts_utc: pd.Timestamp, event_time_et: str, variant: str) -> pd.Timestamp:
    release_ts_utc = pd.Timestamp(release_ts_utc)
    if variant == "macro_plus_120m__vix_next_day_10am":
        return release_ts_utc + pd.Timedelta(minutes=120)
    if variant == "macro_next_grid_after_release__vix_next_day_10am":
        return ceil_hour(release_ts_utc)

    if variant == "macro_market_open_or_30m_buffer__vix_next_day_10am":
        buffer_minutes = 30
    elif variant == "macro_market_open_or_60m_buffer__vix_next_day_10am":
        buffer_minutes = 60
    else:
        raise ValueError(f"Unknown timing variant: {variant}")

    release_ny = release_ts_utc.tz_localize(UTC).tz_convert(NY)
    market_open = ny_market_open_for_date(release_ny.date())
    if release_ny.time() < time(9, 30):
        return market_open
    return release_ts_utc + pd.Timedelta(minutes=buffer_minutes)


def apply_macro_variant_availability(macro: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = macro.copy()
    out["release_ts_utc"] = parse_utc(out["release_ts_utc"])
    out["available_ts_utc"] = [
        macro_available_ts(ts, event_time, variant)
        for ts, event_time in zip(out["release_ts_utc"], out["assumed_release_time_et"])
    ]
    out["timing_variant"] = variant
    return out


def market_open_mask_utc(ts: pd.Series) -> pd.Series:
    local = pd.to_datetime(ts, utc=True).dt.tz_convert(NY)
    t = local.dt.time
    return (local.dt.weekday < 5) & (t >= time(9, 30)) & (t <= time(16, 0))


def compute_metric_block(frame: pd.DataFrame, equity_col: str, return_col: str, ann_factor: float) -> dict[str, Any]:
    data = frame.dropna(subset=[equity_col, return_col]).copy()
    if data.empty:
        raise ValueError("No rows available for metric computation")
    start_ts = pd.Timestamp(data["timestamp"].iloc[0])
    end_ts = pd.Timestamp(data["timestamp"].iloc[-1])
    elapsed_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1e-12)
    start_equity = float(data[equity_col].iloc[0])
    final_equity = float(data[equity_col].iloc[-1])
    returns = data[return_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    cagr = (final_equity / start_equity) ** (365.0 / elapsed_days) - 1.0
    vol = float(returns.std(ddof=1) * math.sqrt(ann_factor)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(ann_factor)) if len(returns) > 1 and returns.std(ddof=1) != 0 else 0.0
    downside = returns[returns < 0]
    sortino = float(returns.mean() / downside.std(ddof=1) * math.sqrt(ann_factor)) if len(downside) > 1 and downside.std(ddof=1) != 0 else 0.0
    dd = data[equity_col].astype(float) / data[equity_col].astype(float).cummax() - 1.0
    return {
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "start_date": start_ts.date().isoformat(),
        "end_date": end_ts.date().isoformat(),
        "obs": int(len(data)),
        "start_equity": start_equity,
        "final_equity": final_equity,
        "total_return": float(final_equity / start_equity - 1.0),
        "cagr": float(cagr),
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()),
    }


def copy_code_to_volume(root: str | os.PathLike[str] | None = None) -> None:
    base = ensure_v4_1_dirs(root)
    code_dir = base / "code"
    here = Path(__file__).resolve().parent
    for py in here.glob("v4_1_*.py"):
        target = assert_v4_1_path(code_dir / py.name, base)
        if py.resolve() == target.resolve():
            continue
        shutil.copy2(py, target)


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
