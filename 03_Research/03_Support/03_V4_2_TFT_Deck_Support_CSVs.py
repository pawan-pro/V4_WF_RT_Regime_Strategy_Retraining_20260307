# Databricks notebook source
"""V4.2 strict-PIT TFT deck-support CSV/Excel package.

Research/deck support only. This script reads isolated MUB-27 research outputs
for the strict-PIT TFT candidate:

`tft72_bil_drawdown_aware_dd_010_ret_005`

and writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_tft_deck_support_2026-05-07/`

It does not write production Delta tables, mutate Databricks workflows/jobs,
touch Saxo/token refresh/RT/dashboard notebooks, write to the model registry, or
edit trading logic.
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

import numpy as np
import pandas as pd


BASE = Path("/Volumes/pawan_quant_research_1/v4/market_data")
SOURCE_ROOT = BASE / "v4_2_label_redesign_strict_pit_2026-05-06"
TFT_WINDOW_ROOT = BASE / "v4_2_tft_window_defensive_sleeve_2026-05-06"
DEFENSIVE_PRICE_ROOT = BASE / "v4_2_defensive_sleeve_prices_2026-05-06"
LEVERAGE_ROOT = BASE / "v4_2_leverage_sensitivity_2026-05-06"
OUTPUT_ROOT = BASE / "v4_2_tft_deck_support_2026-05-07"

VARIANT = "tft72_bil_drawdown_aware_dd_010_ret_005"
DAILY_SOURCE = SOURCE_ROOT / "performance" / f"{VARIANT}_daily_returns.csv"
SIGNAL_SOURCE = SOURCE_ROOT / "signals" / f"{VARIANT}_signals.csv"
COMPARISON_SOURCE = SOURCE_ROOT / "forensics" / "variant_comparison.csv"
LABEL_DISTRIBUTION_SOURCE = SOURCE_ROOT / "forensics" / "label_distribution_summary.csv"
DEFENSIVE_COVERAGE_SOURCE = DEFENSIVE_PRICE_ROOT / "coverage" / "defensive_etf_coverage_summary.csv"
PRIOR_LEVERAGE_SOURCE = LEVERAGE_ROOT / "leverage_summary.csv"

INITIAL_CAPITAL = 5_000_000.0
SPY_CAGR = 0.127546
B_C2_CAGR = 0.208405
TRADING_DAYS = 252
LEVERAGE_GRID = [1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00]
FIXED_FINANCING_PROXY = 0.045
FINANCING_ASSUMPTION = (
    "Fixed 4.50% annual financing proxy used because an exact 10Y Treasury "
    "series was not present in the inspected MUB-27 research artifacts. This is "
    "10Y-ish proxy plus financing premium, documented for deck sensitivity only."
)
EXPECTED = {
    "CAGR": 0.110980,
    "Sharpe": 1.0233,
    "Sortino": 1.1103,
    "Max DD": -0.156828,
    "Final Equity": 8_463_000.0,
    "Switch Count": 137,
}


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def assert_output_path(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside deck-support output root: {path}")


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_ROOT / name
    assert_output_path(path)
    df.to_csv(path, index=False)
    return path


def write_text(text: str, name: str) -> Path:
    path = OUTPUT_ROOT / name
    assert_output_path(path)
    path.write_text(text, encoding="utf-8")
    return path


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


def write_json(payload: dict[str, Any], name: str) -> Path:
    path = OUTPUT_ROOT / name
    assert_output_path(path)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    return pd.read_csv(path)


def normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "report_date" not in out.columns:
        out["report_date"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.date.astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce").dt.date.astype(str)
    for col in ["strategy_daily_return", "spy_daily_return", "strategy_equity", "spy_equity"]:
        if col not in out.columns:
            raise ValueError(f"Daily source missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "strategy_equity", "spy_equity"]).sort_values("timestamp").reset_index(drop=True)
    out["strategy_drawdown"] = out["strategy_equity"] / out["strategy_equity"].cummax() - 1.0
    out["spy_drawdown"] = out["spy_equity"] / out["spy_equity"].cummax() - 1.0
    out["strategy_cumulative_return"] = out["strategy_equity"] / INITIAL_CAPITAL - 1.0
    out["spy_cumulative_return"] = out["spy_equity"] / INITIAL_CAPITAL - 1.0
    return out


def period_performance(daily: pd.DataFrame, freq: str, label_name: str) -> pd.DataFrame:
    work = daily.copy()
    work["period"] = pd.to_datetime(work["report_date"]).dt.to_period(freq).astype(str)
    rows = []
    for period, group in work.groupby("period", sort=True):
        start_strategy = float(group["strategy_equity"].iloc[0])
        end_strategy = float(group["strategy_equity"].iloc[-1])
        start_spy = float(group["spy_equity"].iloc[0])
        end_spy = float(group["spy_equity"].iloc[-1])
        rows.append(
            {
                "period_type": label_name,
                "period": period,
                "start_date": group["report_date"].iloc[0],
                "end_date": group["report_date"].iloc[-1],
                "strategy_return": end_strategy / start_strategy - 1.0,
                "spy_return": end_spy / start_spy - 1.0,
                "excess_return_vs_spy": (end_strategy / start_strategy - 1.0) - (end_spy / start_spy - 1.0),
                "start_equity": start_strategy,
                "end_equity": end_strategy,
                "min_equity": float(group["strategy_equity"].min()),
                "period_max_drawdown": float((group["strategy_equity"] / group["strategy_equity"].cummax() - 1.0).min()),
                "daily_rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def half_yearly_performance(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    dt = pd.to_datetime(work["report_date"])
    half = np.where(dt.dt.month <= 6, "H1", "H2")
    work["period"] = dt.dt.year.astype(str) + "-" + half
    rows = []
    for period, group in work.groupby("period", sort=True):
        start_strategy = float(group["strategy_equity"].iloc[0])
        end_strategy = float(group["strategy_equity"].iloc[-1])
        start_spy = float(group["spy_equity"].iloc[0])
        end_spy = float(group["spy_equity"].iloc[-1])
        rows.append(
            {
                "period_type": "half_year",
                "period": period,
                "start_date": group["report_date"].iloc[0],
                "end_date": group["report_date"].iloc[-1],
                "strategy_return": end_strategy / start_strategy - 1.0,
                "spy_return": end_spy / start_spy - 1.0,
                "excess_return_vs_spy": (end_strategy / start_strategy - 1.0) - (end_spy / start_spy - 1.0),
                "start_equity": start_strategy,
                "end_equity": end_strategy,
                "min_equity": float(group["strategy_equity"].min()),
                "period_max_drawdown": float((group["strategy_equity"] / group["strategy_equity"].cummax() - 1.0).min()),
                "daily_rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def switch_count_from_signals() -> int | None:
    if not SIGNAL_SOURCE.exists():
        return None
    signals = pd.read_csv(SIGNAL_SOURCE)
    if "predicted_regime" not in signals.columns:
        return None
    regimes = pd.to_numeric(signals["predicted_regime"], errors="coerce").dropna().astype(int).to_numpy()
    if len(regimes) < 2:
        return 0
    return int(np.sum(regimes[1:] != regimes[:-1]))


def metrics_from_daily(daily: pd.DataFrame) -> dict[str, Any]:
    ret = daily["strategy_daily_return"].astype(float)
    risk_ret = ret.iloc[1:] if len(ret) > 1 else ret
    downside = risk_ret[risk_ret < 0]
    final_equity = float(daily["strategy_equity"].iloc[-1])
    max_dd = float(daily["strategy_drawdown"].min())
    cagr = (final_equity / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0
    vol = risk_ret.std(ddof=1) * math.sqrt(TRADING_DAYS) if risk_ret.std(ddof=1) else 0.0
    sharpe = risk_ret.mean() / risk_ret.std(ddof=1) * math.sqrt(TRADING_DAYS) if risk_ret.std(ddof=1) else 0.0
    sortino = risk_ret.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS) if len(downside) > 1 and downside.std(ddof=1) else 0.0
    switch_count = None
    if COMPARISON_SOURCE.exists():
        comp = pd.read_csv(COMPARISON_SOURCE)
        match = comp[comp.get("variant", pd.Series(dtype=str)).astype(str).eq(VARIANT)]
        if not match.empty and "Switch Count" in match.columns:
            switch_count = int(match["Switch Count"].iloc[0])
    if switch_count is None:
        switch_count = switch_count_from_signals()
    return {
        "variant": VARIANT,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "Final Equity": final_equity,
        "Switch Count": int(switch_count) if switch_count is not None else np.nan,
        "Annualized Volatility": vol,
        "Gap vs SPY CAGR": cagr - SPY_CAGR,
        "Gap vs B/C2 CAGR": cagr - B_C2_CAGR,
        "daily_rows": len(daily),
        "start_date": daily["report_date"].iloc[0],
        "end_date": daily["report_date"].iloc[-1],
    }


def leverage_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    base_returns = daily["strategy_daily_return"].astype(float).to_numpy()
    rows = []
    for lev in LEVERAGE_GRID:
        financing_daily = max(lev - 1.0, 0.0) * FIXED_FINANCING_PROXY / TRADING_DAYS
        levered_returns = lev * base_returns - financing_daily
        factors = 1.0 + levered_returns
        breached_zero = bool((factors <= 0.0).any())
        equity = INITIAL_CAPITAL * pd.Series(factors).cumprod()
        if breached_zero:
            first_bad = int(np.flatnonzero(factors <= 0.0)[0])
            equity.iloc[first_bad:] = 0.0
            levered_returns[first_bad] = -1.0
            if first_bad + 1 < len(levered_returns):
                levered_returns[first_bad + 1 :] = 0.0
        ret = pd.Series(levered_returns)
        risk_ret = ret.iloc[1:] if len(ret) > 1 else ret
        downside = risk_ret[risk_ret < 0]
        final_equity = float(equity.iloc[-1])
        dd = equity / equity.cummax().replace(0.0, np.nan) - 1.0
        max_dd = float(dd.fillna(-1.0 if breached_zero else 0.0).min())
        cagr = -1.0 if breached_zero else (final_equity / INITIAL_CAPITAL) ** (1.0 / 5.0) - 1.0
        rows.append(
            {
                "variant": VARIANT,
                "leverage": lev,
                "financing_assumption": FINANCING_ASSUMPTION,
                "annual_financing_rate_proxy": FIXED_FINANCING_PROXY,
                "CAGR": cagr,
                "Sharpe": risk_ret.mean() / risk_ret.std(ddof=1) * math.sqrt(TRADING_DAYS) if risk_ret.std(ddof=1) else 0.0,
                "Sortino": risk_ret.mean() / downside.std(ddof=1) * math.sqrt(TRADING_DAYS) if len(downside) > 1 and downside.std(ddof=1) else 0.0,
                "Max DD": max_dd,
                "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
                "Final Equity": final_equity,
                "Annualized Volatility": risk_ret.std(ddof=1) * math.sqrt(TRADING_DAYS) if risk_ret.std(ddof=1) else 0.0,
                "worst_daily_return": float(risk_ret.min()) if len(risk_ret) else np.nan,
                "best_daily_return": float(risk_ret.max()) if len(risk_ret) else np.nan,
                "breached_zero": breached_zero,
                "risk_flag": "failed" if breached_zero else "severe" if max_dd <= -0.80 else "high" if max_dd <= -0.60 else "normal",
            }
        )
    return pd.DataFrame(rows)


def reconciliation(metrics: dict[str, Any]) -> pd.DataFrame:
    tolerances = {
        "CAGR": 0.0001,
        "Sharpe": 0.0003,
        "Sortino": 0.0005,
        "Max DD": 0.0002,
        "Final Equity": 25_000.0,
        "Switch Count": 0.0,
    }
    rows = []
    for metric, expected in EXPECTED.items():
        observed = metrics.get(metric)
        diff = observed - expected if pd.notna(observed) else np.nan
        tol = tolerances[metric]
        rows.append(
            {
                "metric": metric,
                "observed": observed,
                "expected": expected,
                "difference": diff,
                "tolerance": tol,
                "passes": bool(abs(diff) <= tol) if pd.notna(diff) else False,
            }
        )
    return pd.DataFrame(rows)


def presentation_numbers(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Candidate", VARIANT, "Strict-PIT TFT 72h + BIL + drawdown-aware label"),
        ("CAGR", metrics["CAGR"], "5-year CAGR, 2021-2025"),
        ("Sharpe", metrics["Sharpe"], "Daily returns, 252 annualization, rf=0"),
        ("Sortino", metrics["Sortino"], "Daily downside deviation, 252 annualization"),
        ("Max Drawdown", metrics["Max DD"], "Daily-last equity drawdown"),
        ("Calmar", metrics["Calmar"], "CAGR / abs(Max DD)"),
        ("Final Equity", metrics["Final Equity"], "Initial capital $5.0M"),
        ("Switches", metrics["Switch Count"], "Regime/allocation switch count from source run"),
        ("Gap vs SPY CAGR", metrics["Gap vs SPY CAGR"], "Candidate CAGR minus SPY CAGR"),
        ("Gap vs B/C2 CAGR", metrics["Gap vs B/C2 CAGR"], "Candidate CAGR minus non-PIT B/C2 reference CAGR"),
        ("Leverage assumption", FINANCING_ASSUMPTION, "For leverage_sensitivity.csv only"),
    ]
    return pd.DataFrame(rows, columns=["item", "value", "note"])


def summary_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    ordered = [
        "variant",
        "CAGR",
        "Sharpe",
        "Sortino",
        "Max DD",
        "Calmar",
        "Final Equity",
        "Switch Count",
        "Annualized Volatility",
        "Gap vs SPY CAGR",
        "Gap vs B/C2 CAGR",
        "daily_rows",
        "start_date",
        "end_date",
    ]
    return pd.DataFrame([{"metric": key, "value": metrics.get(key)} for key in ordered])


def source_hashes() -> pd.DataFrame:
    paths = {
        "daily_source": DAILY_SOURCE,
        "signal_source": SIGNAL_SOURCE,
        "comparison_source": COMPARISON_SOURCE,
        "label_distribution_source": LABEL_DISTRIBUTION_SOURCE,
        "defensive_coverage_source": DEFENSIVE_COVERAGE_SOURCE,
        "prior_leverage_source_for_context_only": PRIOR_LEVERAGE_SOURCE,
    }
    return pd.DataFrame(
        [
            {
                "source_name": name,
                "path": str(path),
                "exists": path.exists(),
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        ]
    )


def write_excel_package(outputs: dict[str, pd.DataFrame]) -> Path | None:
    path = OUTPUT_ROOT / "v4_2_tft_deck_support.xlsx"
    assert_output_path(path)
    try:
        try:
            import openpyxl  # noqa: F401
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openpyxl"])
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet, df in outputs.items():
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
        return path
    except Exception as exc:
        write_text(f"Excel package was not created because pandas ExcelWriter failed: {exc}\n", "excel_generation_error.txt")
        return None


def write_readme(metrics: dict[str, Any], rec: pd.DataFrame, generated: list[str], excel_path: Path | None) -> None:
    lines = [
        "# V4.2 Strict-PIT TFT Deck Support",
        "",
        "Research/deck support package for `tft72_bil_drawdown_aware_dd_010_ret_005`.",
        "",
        "This package is for the V4.2 strict-PIT TFT candidate. It must not be mixed with the V4.1/B-C2 XGBoost teacher-smoothed leverage-analysis Excel.",
        "",
        "## Leverage Assumption",
        "",
        FINANCING_ASSUMPTION,
        "",
        "## Reconciled Metrics",
        "",
        rec.to_csv(index=False),
        "",
        "## Generated Files",
        "",
        *[f"- `{name}`" for name in generated],
        "",
        f"- Excel workbook: `{excel_path.name}`" if excel_path else "- Excel workbook: not generated; see `excel_generation_error.txt` if present.",
        "",
        "## Production Safety",
        "",
        "No production V4 Delta tables, Saxo execution, token refresh, RT/dashboard notebooks, workflows, model registry entries, or trading logic were changed.",
        "",
    ]
    write_text("\n".join(lines), "README.md")


def main() -> None:
    ensure_dirs()
    daily = normalize_daily(read_csv(DAILY_SOURCE))
    metrics = metrics_from_daily(daily)
    daily_out = daily[
        [
            "timestamp",
            "report_date",
            "strategy_daily_return",
            "spy_daily_return",
            "strategy_equity",
            "spy_equity",
            "strategy_drawdown",
            "spy_drawdown",
            "strategy_cumulative_return",
            "spy_cumulative_return",
        ]
    ].copy()
    monthly = period_performance(daily, "M", "month")
    quarterly = period_performance(daily, "Q", "quarter")
    half_yearly = half_yearly_performance(daily)
    yearly = period_performance(daily, "Y", "year")
    summary = summary_metrics(metrics)
    pres = presentation_numbers(metrics)
    leverage = leverage_sensitivity(daily)
    rec = reconciliation(metrics)
    hashes = source_hashes()
    if not bool(rec["passes"].all()):
        print("WARNING: metric reconciliation did not fully pass")
        print(rec.to_string(index=False))

    outputs = {
        "daily_performance": daily_out,
        "monthly_performance": monthly,
        "quarterly_performance": quarterly,
        "half_yearly_performance": half_yearly,
        "yearly_performance": yearly,
        "summary_metrics": summary,
        "presentation_numbers": pres,
        "leverage_sensitivity": leverage,
        "metric_reconciliation": rec,
        "source_files": hashes,
    }
    generated = []
    for name, df in outputs.items():
        csv_name = f"{name}.csv"
        write_csv(df, csv_name)
        generated.append(csv_name)
    excel_path = write_excel_package(outputs)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "output_root": OUTPUT_ROOT,
        "variant": VARIANT,
        "source_root": SOURCE_ROOT,
        "generated_files": generated + ([excel_path.name] if excel_path else []),
        "leverage_assumption": FINANCING_ASSUMPTION,
        "metric_reconciliation_passes": bool(rec["passes"].all()),
        "production_delta_writes": False,
        "workflow_or_schedule_changes": False,
        "saxo_or_token_changes": False,
        "rt_or_dashboard_changes": False,
        "model_registry_writes": False,
        "trading_logic_changes": False,
    }
    write_json(manifest, "runtime_manifest.json")
    write_readme(metrics, rec, generated, excel_path)
    print(json.dumps(jsonable(manifest), indent=2, sort_keys=True))
    print(rec.to_string(index=False))


if __name__ == "__main__":
    main()
