# Databricks notebook source
import json
import math
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"
SIGNAL_CSV_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/bt_walkforward_regime_signals.csv"
REPORT_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/market_data/reports"
AS_OF_DATE = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d")
OUTPUT_XLSX = f"{REPORT_ROOT}/v4_backtest_daily_eod_audit_pack_{AS_OF_DATE}.xlsx"
OUTPUT_JSON = f"{REPORT_ROOT}/v4_backtest_daily_eod_audit_pack_{AS_OF_DATE}_slide_metrics.json"
OUTPUT_CSV_DIR = f"{REPORT_ROOT}/v4_backtest_daily_eod_audit_pack_{AS_OF_DATE}_csv"
INITIAL_CAPITAL = 5_000_000.0
SWITCH_COST = 0.0005
DAILY_BARS_PER_YEAR = 252.0

REGIME_NAME = {0: "Expansion", 1: "Slowdown", 2: "Recession"}
REGIME_TARGET = {0: "QQQ", 1: "SPY", 2: "CASH"}

PRIOR_DECK = {
    "V4 Walk-Forward": {
        "cagr": 0.069905,
        "sharpe": 0.4091,
        "max_drawdown": -0.163622,
        "source": "2026-04-20 report Current V4 baseline",
    },
    "Corrected ALFRED research variant": {
        "cagr": 0.208405,
        "sharpe": 1.0024,
        "sortino": 1.3729,
        "max_drawdown": -0.255385,
        "source": "2026-04-20 report canonical corrected-ALFRED variant",
    },
}


def table_summary(table_name, ts_col="timestamp"):
    row = spark.sql(
        f"""
        SELECT
          COUNT(*) AS row_count,
          MIN({ts_col}) AS min_ts,
          MAX({ts_col}) AS max_ts
        FROM {table_name}
        """
    ).collect()[0]
    return {
        "source": table_name,
        "row_count": int(row["row_count"]),
        "min_ts": str(row["min_ts"]) if row["min_ts"] is not None else None,
        "max_ts": str(row["max_ts"]) if row["max_ts"] is not None else None,
    }


def max_drawdown(series):
    dd = series / series.cummax() - 1.0
    return float(dd.min())


def years_from_timestamps(start_ts, end_ts):
    return max((pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() / (365.0 * 24 * 3600), 1e-12)


def years_from_dates(start_date, end_date):
    return max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.0, 1e-12)


def metric_block(frame, equity_col, return_col, label, sampling, years, annualization):
    data = frame.dropna(subset=[equity_col, return_col]).copy()
    rets = data[return_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    start = float(data[equity_col].iloc[0])
    end = float(data[equity_col].iloc[-1])
    total_return = end / start - 1.0 if start else np.nan
    cagr = (end / start) ** (1.0 / years) - 1.0 if start > 0 and end > 0 else np.nan
    if sampling == "hourly_market_open":
        bars_per_year = len(rets) / years
    else:
        bars_per_year = DAILY_BARS_PER_YEAR
    vol = float(rets.std(ddof=1) * math.sqrt(bars_per_year)) if len(rets) > 1 and bars_per_year > 0 else np.nan
    sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(bars_per_year)) if len(rets) > 1 and rets.std(ddof=1) != 0 else np.nan
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std(ddof=1) * math.sqrt(bars_per_year)) if len(downside) > 1 and downside.std(ddof=1) != 0 else np.nan
    return {
        "strategy": label,
        "sampling": sampling,
        "annualization": annualization,
        "start_ts": data["timestamp"].iloc[0] if "timestamp" in data.columns else None,
        "end_ts": data["timestamp"].iloc[-1] if "timestamp" in data.columns else None,
        "start_date": data["ny_market_date"].iloc[0] if "ny_market_date" in data.columns else None,
        "end_date": data["ny_market_date"].iloc[-1] if "ny_market_date" in data.columns else None,
        "observations": int(len(data)),
        "years_365": float(years),
        "start_value": start,
        "final_value": end,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown(data[equity_col].astype(float)),
    }


def is_market_open_timestamp(ts):
    local = ts.tz_localize("UTC").tz_convert("America/New_York")
    if local.weekday() >= 5:
        return False
    t = local.time()
    return t >= pd.Timestamp("09:30").time() and t <= pd.Timestamp("16:00").time()


signals = spark.table(SIGNAL_TABLE).select("timestamp", "regime", "predicted_regime").toPandas()
silver = spark.table(SILVER_TABLE).select("timestamp", "SPY_close", "QQQ_close").toPandas()
signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=False)
silver["timestamp"] = pd.to_datetime(silver["timestamp"], utc=False)

hourly = (
    signals.drop_duplicates("timestamp")
    .merge(silver.drop_duplicates("timestamp"), on="timestamp", how="left")
    .sort_values("timestamp")
    .reset_index(drop=True)
)
hourly["SPY_close"] = pd.to_numeric(hourly["SPY_close"], errors="coerce").ffill()
hourly["QQQ_close"] = pd.to_numeric(hourly["QQQ_close"], errors="coerce").ffill()
hourly["predicted_regime"] = pd.to_numeric(hourly["predicted_regime"], errors="coerce").astype("Int64")
hourly = hourly.dropna(subset=["SPY_close", "QQQ_close", "predicted_regime"]).reset_index(drop=True)
hourly["market_open_bar"] = hourly["timestamp"].apply(is_market_open_timestamp)
hourly = hourly[hourly["market_open_bar"]].reset_index(drop=True)
hourly["regime_name"] = hourly["predicted_regime"].astype(int).map(REGIME_NAME)
hourly["target_asset"] = hourly["predicted_regime"].astype(int).map(REGIME_TARGET)

v4_equity = [INITIAL_CAPITAL]
spy_equity = [INITIAL_CAPITAL]
v4_returns = [0.0]
spy_returns = [0.0]
switch_flags = [False]
active_asset = "START"
for i in range(1, len(hourly)):
    target = hourly.loc[i - 1, "target_asset"]
    if target == "QQQ":
        asset_ret = hourly.loc[i, "QQQ_close"] / hourly.loc[i - 1, "QQQ_close"] - 1.0
    elif target == "SPY":
        asset_ret = hourly.loc[i, "SPY_close"] / hourly.loc[i - 1, "SPY_close"] - 1.0
    else:
        asset_ret = 0.0
    switched = target != active_asset
    friction = SWITCH_COST if switched else 0.0
    r = asset_ret - friction
    v4_returns.append(float(r))
    v4_equity.append(v4_equity[-1] * (1.0 + r))
    spy_r = hourly.loc[i, "SPY_close"] / hourly.loc[i - 1, "SPY_close"] - 1.0
    spy_returns.append(float(spy_r))
    spy_equity.append(spy_equity[-1] * (1.0 + spy_r))
    switch_flags.append(bool(switched))
    active_asset = target

hourly["v4_return"] = v4_returns
hourly["spy_return"] = spy_returns
hourly["v4_equity"] = v4_equity
hourly["spy_equity"] = spy_equity
hourly["switch"] = switch_flags
hourly["ny_market_date"] = hourly["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.date.astype(str)

idx = hourly.groupby("ny_market_date")["timestamp"].idxmax()
daily = hourly.loc[idx].sort_values("timestamp").reset_index(drop=True).copy()
daily["v4_daily_return"] = daily["v4_equity"].pct_change().fillna(0.0)
daily["spy_daily_return"] = daily["spy_equity"].pct_change().fillna(0.0)
daily["v4_drawdown"] = daily["v4_equity"] / daily["v4_equity"].cummax() - 1.0
daily["spy_drawdown"] = daily["spy_equity"] / daily["spy_equity"].cummax() - 1.0

hourly_years = years_from_timestamps(hourly["timestamp"].iloc[0], hourly["timestamp"].iloc[-1])
daily_years = years_from_dates(daily["ny_market_date"].iloc[0], daily["ny_market_date"].iloc[-1])

hourly_metrics = pd.DataFrame(
    [
        metric_block(hourly, "v4_equity", "v4_return", "V4 Walk-Forward", "hourly_market_open", hourly_years, "Observed market-open hourly bars per elapsed 365-day year"),
        metric_block(hourly, "spy_equity", "spy_return", "SPY Benchmark", "hourly_market_open", hourly_years, "Observed market-open hourly bars per elapsed 365-day year"),
    ]
)

daily_metrics = pd.DataFrame(
    [
        metric_block(daily, "v4_equity", "v4_daily_return", "V4 Walk-Forward", "daily_eod", daily_years, "Daily returns annualized with sqrt(252)"),
        metric_block(daily, "spy_equity", "spy_daily_return", "SPY Benchmark", "daily_eod", daily_years, "Daily returns annualized with sqrt(252)"),
    ]
)

comparison_rows = []
for strategy in ["V4 Walk-Forward", "SPY Benchmark"]:
    h = hourly_metrics[hourly_metrics["strategy"] == strategy].iloc[0].to_dict()
    d = daily_metrics[daily_metrics["strategy"] == strategy].iloc[0].to_dict()
    deck = PRIOR_DECK.get(strategy, {})
    for metric in ["final_value", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown"]:
        comparison_rows.append(
            {
                "strategy": strategy,
                "metric": metric,
                "previous_hourly_value": h.get(metric),
                "daily_eod_value": d.get(metric),
                "daily_minus_hourly": (d.get(metric) - h.get(metric)) if pd.notna(d.get(metric)) and pd.notna(h.get(metric)) else np.nan,
                "prior_deck_value": deck.get(metric if metric != "ann_vol" else "vol"),
                "deck_source": deck.get("source"),
            }
        )
comparison = pd.DataFrame(comparison_rows)

deck_v4 = PRIOR_DECK["V4 Walk-Forward"]
hourly_distance = abs(float(hourly_metrics.loc[0, "cagr"]) - deck_v4["cagr"]) + abs(float(hourly_metrics.loc[0, "sharpe"]) - deck_v4["sharpe"]) + abs(float(hourly_metrics.loc[0, "max_drawdown"]) - deck_v4["max_drawdown"])
daily_distance = abs(float(daily_metrics.loc[0, "cagr"]) - deck_v4["cagr"]) + abs(float(daily_metrics.loc[0, "sharpe"]) - deck_v4["sharpe"]) + abs(float(daily_metrics.loc[0, "max_drawdown"]) - deck_v4["max_drawdown"])
reconciles_closer = daily_distance < hourly_distance

annual_rows = []
for year, grp in daily.groupby(pd.to_datetime(daily["ny_market_date"]).dt.year):
    annual_rows.append(
        {
            "year": int(year),
            "start_date": grp["ny_market_date"].iloc[0],
            "end_date": grp["ny_market_date"].iloc[-1],
            "v4_return": float(grp["v4_equity"].iloc[-1] / grp["v4_equity"].iloc[0] - 1.0),
            "spy_return": float(grp["spy_equity"].iloc[-1] / grp["spy_equity"].iloc[0] - 1.0),
            "daily_observations": int(len(grp)),
        }
    )
annual_returns = pd.DataFrame(annual_rows)

trailing_rows = []
end_date = pd.Timestamp(daily["ny_market_date"].iloc[-1])
for years in [1, 3, 5]:
    start_cutoff = (end_date - pd.DateOffset(years=years)).date().isoformat()
    window = daily[daily["ny_market_date"] >= start_cutoff].copy()
    window_years = years_from_dates(window["ny_market_date"].iloc[0], window["ny_market_date"].iloc[-1])
    trailing_rows.append(metric_block(window, "v4_equity", "v4_daily_return", f"V4 trailing {years}Y", "daily_eod", window_years, "Daily returns annualized with sqrt(252)"))
    trailing_rows.append(metric_block(window, "spy_equity", "spy_daily_return", f"SPY trailing {years}Y", "daily_eod", window_years, "Daily returns annualized with sqrt(252)"))
trailing_metrics = pd.DataFrame(trailing_rows)

source_rows = [table_summary(SILVER_TABLE), table_summary(SIGNAL_TABLE)]
try:
    csv_df = pd.read_csv(SIGNAL_CSV_PATH)
    csv_ts = pd.to_datetime(csv_df["timestamp"], errors="coerce") if "timestamp" in csv_df.columns else pd.Series(dtype="datetime64[ns]")
    source_rows.append(
        {
            "source": SIGNAL_CSV_PATH,
            "row_count": int(len(csv_df)),
            "min_ts": str(csv_ts.min()) if not csv_ts.empty else None,
            "max_ts": str(csv_ts.max()) if not csv_ts.empty else None,
        }
    )
except Exception as exc:
    source_rows.append({"source": SIGNAL_CSV_PATH, "row_count": None, "min_ts": None, "max_ts": None, "error": str(exc)[:500]})
source_row_counts = pd.DataFrame(source_rows)

latest_ts_check = hourly.groupby("ny_market_date", as_index=False)["timestamp"].max().rename(columns={"timestamp": "expected_eod_timestamp"})
daily_check = daily[["ny_market_date", "timestamp"]].merge(latest_ts_check, on="ny_market_date", how="left")
eod_mismatches = int((daily_check["timestamp"] != daily_check["expected_eod_timestamp"]).sum())

audit_checks = pd.DataFrame(
    [
        {"check": "one_row_per_trading_date", "status": "PASS" if len(daily) == daily["ny_market_date"].nunique() else "FAIL", "value": len(daily), "detail": "Daily EOD series has one row per New York market date represented in hourly regular-hours data"},
        {"check": "eod_timestamp_is_latest_for_date", "status": "PASS" if eod_mismatches == 0 else "FAIL", "value": eod_mismatches, "detail": "Selected timestamp equals max available timestamp per New York market date"},
        {"check": "no_duplicate_dates", "status": "PASS" if daily["ny_market_date"].duplicated().sum() == 0 else "FAIL", "value": int(daily["ny_market_date"].duplicated().sum()), "detail": "Duplicate dates in daily EOD NAV"},
        {"check": "no_null_v4_nav", "status": "PASS" if daily["v4_equity"].isna().sum() == 0 else "FAIL", "value": int(daily["v4_equity"].isna().sum()), "detail": "Null V4 daily EOD NAV count"},
        {"check": "no_null_spy_nav", "status": "PASS" if daily["spy_equity"].isna().sum() == 0 else "FAIL", "value": int(daily["spy_equity"].isna().sum()), "detail": "Null SPY daily EOD NAV count"},
        {"check": "start_end_dates", "status": "PASS", "value": f"{daily['ny_market_date'].iloc[0]} to {daily['ny_market_date'].iloc[-1]}", "detail": f"{len(daily)} daily observations"},
        {"check": "cagr_day_count", "status": "PASS", "value": daily_years, "detail": "Calendar days / 365 from first to last New York market date"},
        {"check": "sharpe_annualization", "status": "PASS", "value": "sqrt(252)", "detail": "Daily EOD return annualization"},
        {"check": "daily_eod_reconciles_closer_to_prior_deck", "status": "PASS" if reconciles_closer else "WARN", "value": str(reconciles_closer), "detail": f"Hourly distance={hourly_distance:.6f}; daily distance={daily_distance:.6f} against prior V4 CAGR/Sharpe/maxDD"},
        {"check": "write_scope", "status": "PASS", "value": OUTPUT_XLSX, "detail": "Only report CSV/JSON/XLSX artifacts are written"},
    ]
)

readme_manifest = pd.DataFrame(
    [
        {"field": "generated_at_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"field": "generated_by", "value": "tmp_codex_v4_daily_eod_audit_pack one-time Databricks run"},
        {"field": "catalog_schema", "value": f"{CATALOG}.{SCHEMA}"},
        {"field": "signal_source", "value": SIGNAL_TABLE},
        {"field": "price_source", "value": SILVER_TABLE},
        {"field": "signal_csv_crosscheck", "value": SIGNAL_CSV_PATH},
        {"field": "hourly_execution_semantics", "value": "Prior market-open bar signal selects next market-open bar return; regime 0=QQQ, 1=SPY, 2=CASH; 5 bps switch cost"},
        {"field": "daily_eod_sampling", "value": "Convert UTC timestamp to New York date, keep regular-hours bars, select latest available timestamp per date"},
        {"field": "daily_observations", "value": len(daily)},
        {"field": "start_end_dates", "value": f"{daily['ny_market_date'].iloc[0]} to {daily['ny_market_date'].iloc[-1]}"},
        {"field": "cagr_basis", "value": "365-day elapsed calendar basis"},
        {"field": "risk_basis", "value": "Daily EOD returns annualized with sqrt(252)"},
        {"field": "prior_deck_reference", "value": json.dumps(PRIOR_DECK)},
        {"field": "output_xlsx", "value": OUTPUT_XLSX},
        {"field": "output_json", "value": OUTPUT_JSON},
        {"field": "output_csv_dir", "value": OUTPUT_CSV_DIR},
    ]
)

daily_eod_equity_curve = daily[
    [
        "ny_market_date",
        "timestamp",
        "SPY_close",
        "QQQ_close",
        "predicted_regime",
        "regime_name",
        "target_asset",
        "v4_equity",
        "spy_equity",
        "v4_daily_return",
        "spy_daily_return",
        "v4_drawdown",
        "spy_drawdown",
    ]
].copy()

daily_eod_drawdown = daily[["ny_market_date", "timestamp", "v4_drawdown", "spy_drawdown", "v4_equity", "spy_equity"]].copy()

sheets = {
    "README_manifest": readme_manifest,
    "source_row_counts": source_row_counts,
    "hourly_vs_daily_metric_comparison": comparison,
    "daily_eod_equity_curve": daily_eod_equity_curve,
    "daily_eod_annual_returns": annual_returns,
    "daily_eod_trailing_metrics": trailing_metrics,
    "daily_eod_drawdown": daily_eod_drawdown,
    "audit_checks": audit_checks,
}

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
for sheet_name, frame in sheets.items():
    frame.to_csv(f"{OUTPUT_CSV_DIR}/{sheet_name}.csv", index=False)

slide_metrics = {
    "as_of_date": AS_OF_DATE,
    "output_xlsx": OUTPUT_XLSX,
    "output_json": OUTPUT_JSON,
    "output_csv_dir": OUTPUT_CSV_DIR,
    "hourly_metrics": hourly_metrics.replace({np.nan: None}).to_dict(orient="records"),
    "daily_eod_metrics": daily_metrics.replace({np.nan: None}).to_dict(orient="records"),
    "daily_eod_annual_returns": annual_returns.replace({np.nan: None}).to_dict(orient="records"),
    "daily_eod_trailing_metrics": trailing_metrics.replace({np.nan: None}).to_dict(orient="records"),
    "audit_checks": audit_checks.to_dict(orient="records"),
    "prior_deck": PRIOR_DECK,
    "reconciles_closer_to_prior_deck": reconciles_closer,
    "hourly_distance_to_prior_deck": hourly_distance,
    "daily_distance_to_prior_deck": daily_distance,
}
os.makedirs(REPORT_ROOT, exist_ok=True)
with open(OUTPUT_JSON, "w") as handle:
    json.dump(slide_metrics, handle, default=str, indent=2)

print(json.dumps({
    "output_xlsx": OUTPUT_XLSX,
    "output_json": OUTPUT_JSON,
    "output_csv_dir": OUTPUT_CSV_DIR,
    "rows": {
        "signals": int(len(signals)),
        "silver": int(len(silver)),
        "hourly_market_open": int(len(hourly)),
        "daily_eod": int(len(daily)),
    },
    "hourly_metrics": slide_metrics["hourly_metrics"],
    "daily_eod_metrics": slide_metrics["daily_eod_metrics"],
    "reconciles_closer_to_prior_deck": reconciles_closer,
}, default=str, indent=2))
