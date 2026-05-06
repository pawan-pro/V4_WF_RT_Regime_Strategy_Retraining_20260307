# Databricks notebook source
"""MUB-27 V4.2 defensive ETF Alpaca price pull.

Research-only. Fetches hourly Alpaca prices for defensive sleeve candidates and
writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_defensive_sleeve_prices_2026-05-06/`

This script never writes or overwrites production Delta tables, including
`pawan_quant_research_1.v4.bronze_bt_market_1h`, and does not mutate Saxo,
token refresh, RT/dashboard notebooks, workflows, model registry, or trading
logic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
OUTPUT_ROOT = BASE / "v4_2_defensive_sleeve_prices_2026-05-06"
PRICE_DIR = OUTPUT_ROOT / "prices"
COVERAGE_DIR = OUTPUT_ROOT / "coverage"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"

FAITHFUL_GRID_PATH = (
    BASE
    / "v4_1_csv"
    / "full_legacy_grid_clone_2026-04-30"
    / "grids"
    / "full_legacy_grid_clone.csv"
)

SYMBOLS = ["BIL", "SHV", "SGOV", "IEF", "AGG", "TLT"]
CASH_LIKE = {"BIL", "SHV", "SGOV"}
DURATION = {"IEF", "AGG", "TLT"}
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
SECRET_SCOPE = "alpaca"
SECRET_KEYS = {"api_key": "api_key_id", "api_secret": "api_secret_key"}
MIN_USABLE_COVERAGE = 0.995


def ensure_python_deps():
    missing = []
    for module_name, package_name in [("alpaca", "alpaca-py")]:
        try:
            __import__(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    try:
        from alpaca.data.enums import DataFeed
    except Exception:
        DataFeed = None

    return StockHistoricalDataClient, StockBarsRequest, TimeFrame, DataFeed


StockHistoricalDataClient, StockBarsRequest, TimeFrame, DataFeed = ensure_python_deps()


def ensure_dirs() -> None:
    for path in [PRICE_DIR, COVERAGE_DIR, MANIFESTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_output(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside research output root: {path}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_text(text: str, path: Path) -> None:
    assert_output(path)
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
    assert_output(path)
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


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def get_dbutils():
    try:
        return dbutils  # type: ignore[name-defined]
    except Exception:
        pass
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        return DBUtils(spark)
    except Exception:
        return None


def get_alpaca_credentials() -> tuple[str | None, str | None, list[str]]:
    dbutils_obj = get_dbutils()
    missing = []
    if dbutils_obj is None:
        return None, None, [f"{SECRET_SCOPE}/{SECRET_KEYS['api_key']}", f"{SECRET_SCOPE}/{SECRET_KEYS['api_secret']}"]
    try:
        api_key = dbutils_obj.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEYS["api_key"])
    except Exception:
        api_key = None
        missing.append(f"{SECRET_SCOPE}/{SECRET_KEYS['api_key']}")
    try:
        api_secret = dbutils_obj.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEYS["api_secret"])
    except Exception:
        api_secret = None
        missing.append(f"{SECRET_SCOPE}/{SECRET_KEYS['api_secret']}")
    return api_key, api_secret, missing


def stop_for_missing_secrets(missing: list[str]) -> None:
    empty_hourly = pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
    empty_daily = pd.DataFrame(columns=["symbol", "date", "timestamp", "close"])
    coverage = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "first_timestamp": None,
                "last_timestamp": None,
                "row_count": 0,
                "trading_days": 0,
                "reference_spy_qqq_price_update_rows": None,
                "exact_aligned_rows": 0,
                "ffill_aligned_rows": 0,
                "coverage_2021_2025": 0.0,
                "exact_missing_rate_vs_spy_qqq_price_update_rows": 1.0,
                "ffill_missing_rate_vs_spy_qqq_price_update_rows": 1.0,
                "classification": "unavailable",
                "reason": "missing Databricks Alpaca secrets: " + ", ".join(missing),
                "recommended_first_sleeve": False,
            }
            for symbol in SYMBOLS
        ]
    )
    alignment = coverage[
        [
            "symbol",
            "reference_spy_qqq_price_update_rows",
            "exact_aligned_rows",
            "ffill_aligned_rows",
            "exact_missing_rate_vs_spy_qqq_price_update_rows",
            "ffill_missing_rate_vs_spy_qqq_price_update_rows",
        ]
    ].rename(
        columns={
            "reference_spy_qqq_price_update_rows": "reference_rows",
            "exact_missing_rate_vs_spy_qqq_price_update_rows": "exact_missing_rate",
            "ffill_missing_rate_vs_spy_qqq_price_update_rows": "ffill_missing_rate",
        }
    )
    write_csv(empty_hourly, PRICE_DIR / "defensive_etf_hourly_prices.csv")
    write_csv(empty_daily, PRICE_DIR / "defensive_etf_daily_last_prices.csv")
    write_csv(coverage, COVERAGE_DIR / "defensive_etf_coverage_summary.csv")
    write_csv(alignment, COVERAGE_DIR / "alignment_vs_spy_qqq_price_update_rows.csv")
    write_json(
        {
            "faithful_legacy_grid": {
                "path": FAITHFUL_GRID_PATH,
                "sha256": file_sha256(FAITHFUL_GRID_PATH),
            },
            "alpaca_request": {
                "symbols": SYMBOLS,
                "start": START,
                "end": END,
                "timeframe": "1Hour",
                "status": "not_fetched_missing_secrets",
            },
        },
        MANIFESTS_DIR / "input_hashes.json",
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "missing_alpaca_secrets",
        "required_secret_refs": missing,
        "output_root": OUTPUT_ROOT,
        "production_delta_writes": False,
        "workflow_or_schedule_changes": False,
    }
    write_json(payload, MANIFESTS_DIR / "runtime_manifest.json")
    write_text(
        "\n".join(
            [
                "# Defensive ETF Alpaca Price Pull",
                "",
                "Run stopped before fetching because required Databricks secrets were unavailable.",
                "",
                "Required secrets:",
                *[f"- `{ref}`" for ref in missing],
                "",
                "No production objects were changed.",
                "",
            ]
        ),
        OUTPUT_ROOT / "README.md",
    )
    print(json.dumps(jsonable(payload), indent=2))


def load_grid_price_update_rows() -> pd.DataFrame:
    grid = normalize_timestamp(pd.read_csv(FAITHFUL_GRID_PATH, usecols=["timestamp", "SPY_close", "QQQ_close"]))
    for col in ["SPY_close", "QQQ_close"]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")
    grid = grid[(grid["timestamp"] >= START.tz_convert(None)) & (grid["timestamp"] <= END.tz_convert(None))].copy()
    grid[["SPY_close", "QQQ_close"]] = grid[["SPY_close", "QQQ_close"]].ffill()
    grid["price_update"] = grid[["SPY_close", "QQQ_close"]].diff().abs().sum(axis=1).gt(0)
    if not grid.empty:
        grid.loc[grid.index[0], "price_update"] = True
    return grid[grid["price_update"]].reset_index(drop=True)


def fetch_bars(client, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = []
    errors = []
    year_starts = pd.date_range(start=start, end=end, freq="YS", tz="UTC")
    if len(year_starts) == 0 or year_starts[0] != start:
        year_starts = pd.DatetimeIndex([start, *list(year_starts[year_starts > start])])
    for symbol in symbols:
        for chunk_start in year_starts:
            chunk_end = min(pd.Timestamp(chunk_start.year, 12, 31, 23, 59, 59, tz="UTC"), end)
            request_kwargs = {
                "symbol_or_symbols": symbol,
                "timeframe": TimeFrame.Hour,
                "start": chunk_start.to_pydatetime(),
                "end": chunk_end.to_pydatetime(),
            }
            if DataFeed is not None:
                request_kwargs["feed"] = DataFeed.IEX
            req = StockBarsRequest(**request_kwargs)
            for attempt in range(3):
                try:
                    bars = client.get_stock_bars(req)
                    if bars.df is not None and not bars.df.empty:
                        frames.append(bars.df.reset_index())
                    break
                except Exception as exc:
                    if attempt == 2:
                        errors.append(
                            {
                                "symbol": symbol,
                                "chunk_start": chunk_start,
                                "chunk_end": chunk_end,
                                "fetch_error": str(exc),
                            }
                        )
                    else:
                        time.sleep(2**attempt)
    if errors:
        write_csv(pd.DataFrame(errors), COVERAGE_DIR / "alpaca_fetch_errors.csv")
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    keep = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    for col in keep:
        if col not in raw.columns:
            raw[col] = np.nan
    out = raw[keep].copy()
    out = normalize_timestamp(out)
    out = out[(out["timestamp"] >= start.tz_convert(None)) & (out["timestamp"] <= end.tz_convert(None))]
    out = out.drop_duplicates(["symbol", "timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


def daily_last_prices(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(columns=["symbol", "date", "timestamp", "close"])
    out = hourly.copy()
    out["date"] = pd.to_datetime(out["timestamp"]).dt.date.astype(str)
    idx = out.groupby(["symbol", "date"])["timestamp"].idxmax()
    return out.loc[idx, ["symbol", "date", "timestamp", "close"]].sort_values(["symbol", "date"]).reset_index(drop=True)


def classify_symbol(symbol: str, coverage: float, first_ts: Any, last_ts: Any, row_count: int) -> tuple[str, str]:
    if row_count == 0 or pd.isna(first_ts) or pd.isna(last_ts):
        return "unavailable", "no Alpaca hourly rows returned"
    if coverage < MIN_USABLE_COVERAGE:
        return "insufficient_history", f"coverage {coverage:.4%} below {MIN_USABLE_COVERAGE:.2%}"
    if symbol in CASH_LIKE:
        return "usable_cash_like", "usable cash-like defensive sleeve"
    if symbol in DURATION:
        return "usable_duration_sleeve", "usable bond-duration defensive sleeve"
    return "usable", "usable"


def coverage_summary(hourly: pd.DataFrame, price_update_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    alignment_rows = []
    ref = price_update_rows[["timestamp"]].copy()
    ref_count = len(ref)
    for symbol in SYMBOLS:
        sym = hourly[hourly["symbol"].eq(symbol)].copy() if not hourly.empty and "symbol" in hourly.columns else pd.DataFrame()
        row_count = int(len(sym))
        first_ts = sym["timestamp"].min() if row_count else None
        last_ts = sym["timestamp"].max() if row_count else None
        trading_days = int(pd.to_datetime(sym["timestamp"]).dt.date.nunique()) if row_count else 0
        aligned = ref.merge(sym[["timestamp", "close"]], on="timestamp", how="left").sort_values("timestamp")
        exact_rows = int(aligned["close"].notna().sum())
        aligned["ffill_close"] = aligned["close"].ffill()
        ffill_rows = int(aligned["ffill_close"].notna().sum())
        exact_missing_rate = 1.0 - exact_rows / ref_count if ref_count else np.nan
        ffill_missing_rate = 1.0 - ffill_rows / ref_count if ref_count else np.nan
        coverage = ffill_rows / ref_count if ref_count else 0.0
        status, reason = classify_symbol(symbol, coverage, first_ts, last_ts, row_count)
        summary_rows.append(
            {
                "symbol": symbol,
                "first_timestamp": first_ts,
                "last_timestamp": last_ts,
                "row_count": row_count,
                "trading_days": trading_days,
                "reference_spy_qqq_price_update_rows": ref_count,
                "exact_aligned_rows": exact_rows,
                "ffill_aligned_rows": ffill_rows,
                "coverage_2021_2025": coverage,
                "exact_missing_rate_vs_spy_qqq_price_update_rows": exact_missing_rate,
                "ffill_missing_rate_vs_spy_qqq_price_update_rows": ffill_missing_rate,
                "classification": status,
                "reason": reason,
                "recommended_first_sleeve": symbol in {"BIL", "SHV"} and status.startswith("usable"),
            }
        )
        alignment_rows.append(
            {
                "symbol": symbol,
                "reference_rows": ref_count,
                "exact_aligned_rows": exact_rows,
                "ffill_aligned_rows": ffill_rows,
                "exact_missing_rate": exact_missing_rate,
                "ffill_missing_rate": ffill_missing_rate,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(alignment_rows)


def write_readme(summary: pd.DataFrame, output_root: Path) -> None:
    usable = summary[summary["classification"].str.startswith("usable")]["symbol"].tolist()
    excluded = summary[~summary["classification"].str.startswith("usable")][["symbol", "classification", "reason"]]
    lines = [
        "# Defensive ETF Alpaca Price Pull",
        "",
        "Research-only hourly price pull for V4.2 defensive sleeve ablations.",
        "",
        f"- Output root: `{output_root}`",
        f"- Period: `{START}` to `{END}`",
        "- Production writes: none",
        "- Production Delta tables/workflows/Saxo/token/model registry/trading logic: unchanged",
        "",
        "## Usable Sleeves",
        "",
        ", ".join(usable) if usable else "None",
        "",
        "## Excluded",
        "",
        excluded.to_csv(index=False) if not excluded.empty else "None",
        "",
    ]
    write_text("\n".join(lines), OUTPUT_ROOT / "README.md")


def main() -> None:
    ensure_dirs()
    api_key, api_secret, missing = get_alpaca_credentials()
    if missing:
        stop_for_missing_secrets(missing)
        return

    client = StockHistoricalDataClient(api_key, api_secret)
    price_update_rows = load_grid_price_update_rows()
    hourly = fetch_bars(client, SYMBOLS, START, END)
    daily = daily_last_prices(hourly)
    summary, alignment = coverage_summary(hourly, price_update_rows)

    write_csv(hourly, PRICE_DIR / "defensive_etf_hourly_prices.csv")
    write_csv(daily, PRICE_DIR / "defensive_etf_daily_last_prices.csv")
    write_csv(summary, COVERAGE_DIR / "defensive_etf_coverage_summary.csv")
    write_csv(alignment, COVERAGE_DIR / "alignment_vs_spy_qqq_price_update_rows.csv")

    input_hashes = {
        "faithful_legacy_grid": {
            "path": FAITHFUL_GRID_PATH,
            "sha256": file_sha256(FAITHFUL_GRID_PATH),
            "price_update_rows_2021_2025": len(price_update_rows),
        },
        "alpaca_request": {
            "symbols": SYMBOLS,
            "start": START,
            "end": END,
            "timeframe": "1Hour",
            "feed": "IEX" if DataFeed is not None else "alpaca_default",
        },
    }
    write_json(input_hashes, MANIFESTS_DIR / "input_hashes.json")
    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "output_root": OUTPUT_ROOT,
        "symbols": SYMBOLS,
        "usable_symbols": summary[summary["classification"].str.startswith("usable")]["symbol"].tolist(),
        "secret_scope_used": SECRET_SCOPE,
        "secret_keys_used": SECRET_KEYS,
        "production_delta_writes": False,
        "bronze_bt_market_1h_writes": False,
        "saxo_or_rt_writes": False,
        "dashboard_writes": False,
        "workflow_or_schedule_changes": False,
        "model_registry_writes": False,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")
    write_readme(summary, OUTPUT_ROOT)
    print(summary.to_string(index=False))
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
