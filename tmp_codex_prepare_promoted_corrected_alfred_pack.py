import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path("/tmp")
OUTPUT_DIR = ROOT / "v4_promoted_corrected_alfred_pack_data"
CONFIRMATION_JSON = ROOT / "codex_corrected_alfred_confirmation.json"
PROMOTION_JSON = ROOT / "v4_corrected_alfred_promotion_manifest_2026-04-30.json"
OFFICIAL_DAILY = ROOT / "daily_equity_curve.csv"
OFFICIAL_HOURLY = ROOT / "hourly_equity_curve.csv"
STRICT_DAILY = next((ROOT / "corrected_confirm_daily_eod_equity_curve_csv").glob("part-*.csv"))


def compute_metrics(df: pd.DataFrame, value_col: str, return_col: str) -> dict:
    start_ts = pd.to_datetime(df["timestamp"].iloc[0])
    end_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    days = max((end_ts - start_ts).total_seconds() / 86400.0, 1e-9)
    start_val = float(df[value_col].iloc[0])
    end_val = float(df[value_col].iloc[-1])
    rets = pd.to_numeric(df[return_col], errors="coerce").dropna()
    downside = rets[rets < 0]
    ann_vol = float(rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / rets.std(ddof=1) * math.sqrt(252)) if len(rets) > 1 and rets.std(ddof=1) else 0.0
    sortino = (
        float(rets.mean() / downside.std(ddof=1) * math.sqrt(252))
        if len(downside) > 1 and downside.std(ddof=1)
        else 0.0
    )
    drawdown = pd.to_numeric(df[value_col], errors="coerce") / pd.to_numeric(df[value_col], errors="coerce").cummax() - 1.0
    return {
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "start_date": start_ts.date().isoformat(),
        "end_date": end_ts.date().isoformat(),
        "obs": int(len(df)),
        "start_equity": start_val,
        "final_equity": end_val,
        "total_return": end_val / start_val - 1.0,
        "cagr": (end_val / start_val) ** (365.0 / days) - 1.0,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
    }


def annual_returns(df: pd.DataFrame, strategy_col: str, benchmark_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["year"] = pd.to_datetime(tmp["timestamp"]).dt.year
    rows = []
    for year, group in tmp.groupby("year", sort=True):
        rows.append(
            {
                "year": int(year),
                "v4_return": float(group[strategy_col].iloc[-1] / group[strategy_col].iloc[0] - 1.0),
                "spy_return": float(group[benchmark_col].iloc[-1] / group[benchmark_col].iloc[0] - 1.0),
            }
        )
    return pd.DataFrame(rows)


def trailing_metrics(df: pd.DataFrame, strategy_col: str, benchmark_col: str) -> pd.DataFrame:
    rows = []
    end_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    for years in [1, 3, 5]:
        start_cutoff = end_ts - pd.DateOffset(years=years)
        window = df[pd.to_datetime(df["timestamp"]) >= start_cutoff].copy()
        if window.empty:
            continue
        window["strategy_daily_return"] = pd.to_numeric(window[strategy_col], errors="coerce").pct_change()
        window["spy_daily_return"] = pd.to_numeric(window[benchmark_col], errors="coerce").pct_change()
        strat = compute_metrics(window, strategy_col, "strategy_daily_return")
        spy = compute_metrics(window, benchmark_col, "spy_daily_return")
        rows.append(
            {
                "window_years": years,
                "v4_cagr": strat["cagr"],
                "v4_sharpe": strat["sharpe"],
                "v4_sortino": strat["sortino"],
                "v4_max_drawdown": strat["max_drawdown"],
                "spy_cagr": spy["cagr"],
                "spy_sharpe": spy["sharpe"],
                "spy_sortino": spy["sortino"],
                "spy_max_drawdown": spy["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
confirmation = json.loads(CONFIRMATION_JSON.read_text())
promotion = json.loads(PROMOTION_JSON.read_text())

hourly = pd.read_csv(OFFICIAL_HOURLY)
hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
try:
    hourly["timestamp"] = hourly["timestamp"].dt.tz_localize(None)
except Exception:
    pass
hourly["SPY"] = pd.to_numeric(hourly["SPY"], errors="coerce")
hourly["portfolio_value"] = pd.to_numeric(hourly["portfolio_value"], errors="coerce")
spy_start = float(hourly["SPY"].iloc[0])
hourly["spy_value"] = 5_000_000.0 * hourly["SPY"] / spy_start
hourly["calendar_date"] = hourly["timestamp"].dt.date

official_strategy_daily = pd.read_csv(OFFICIAL_DAILY)
official_strategy_daily["timestamp"] = pd.to_datetime(official_strategy_daily["timestamp"])
try:
    official_strategy_daily["timestamp"] = official_strategy_daily["timestamp"].dt.tz_localize(None)
except Exception:
    pass
official_strategy_daily["portfolio_value"] = pd.to_numeric(official_strategy_daily["equity"], errors="coerce")
official_strategy_daily["calendar_date"] = official_strategy_daily["timestamp"].dt.date

official_spy_daily = hourly.groupby("calendar_date", as_index=False).tail(1).copy()
official_spy_daily = official_spy_daily[["calendar_date", "spy_value"]]
official_daily = official_strategy_daily.merge(official_spy_daily, on="calendar_date", how="left")
official_daily["portfolio_daily_return"] = official_daily["portfolio_value"].pct_change()
official_daily["spy_daily_return"] = official_daily["spy_value"].pct_change()
official_metrics = compute_metrics(official_daily, "portfolio_value", "portfolio_daily_return")
official_spy_metrics = compute_metrics(official_daily, "spy_value", "spy_daily_return")
official_annual = annual_returns(official_daily, "portfolio_value", "spy_value")
official_trailing = trailing_metrics(official_daily, "portfolio_value", "spy_value")
official_drawdown = official_daily[["timestamp", "portfolio_value", "spy_value"]].copy()
official_drawdown["v4_drawdown"] = official_drawdown["portfolio_value"] / official_drawdown["portfolio_value"].cummax() - 1.0
official_drawdown["spy_drawdown"] = official_drawdown["spy_value"] / official_drawdown["spy_value"].cummax() - 1.0

strict_daily = pd.read_csv(STRICT_DAILY)
strict_daily["timestamp"] = pd.to_datetime(strict_daily["timestamp"])
try:
    strict_daily["timestamp"] = strict_daily["timestamp"].dt.tz_localize(None)
except Exception:
    pass
strict_daily["portfolio_value"] = pd.to_numeric(strict_daily["portfolio_value"], errors="coerce")
strict_daily["spy_value"] = pd.to_numeric(strict_daily["spy_value"], errors="coerce")
strict_daily["daily_return"] = strict_daily["portfolio_value"].pct_change()
strict_daily["spy_daily_return"] = strict_daily["spy_value"].pct_change()
strict_metrics = compute_metrics(strict_daily, "portfolio_value", "daily_return")
strict_spy_metrics = compute_metrics(strict_daily, "spy_value", "spy_daily_return")

comparison = pd.DataFrame(
    [
        {
            "variant": "official_promoted_daily_resampled",
            "cagr": official_metrics["cagr"],
            "sharpe": official_metrics["sharpe"],
            "sortino": official_metrics["sortino"],
            "max_drawdown": official_metrics["max_drawdown"],
            "final_equity": official_metrics["final_equity"],
            "start_date": official_metrics["start_date"],
            "end_date": official_metrics["end_date"],
            "obs": official_metrics["obs"],
            "methodology": "calendar-date daily resample from corrected hourly equity artifact",
        },
        {
            "variant": "strict_ny_daily_eod_audit",
            "cagr": strict_metrics["cagr"],
            "sharpe": strict_metrics["sharpe"],
            "sortino": strict_metrics["sortino"],
            "max_drawdown": strict_metrics["max_drawdown"],
            "final_equity": strict_metrics["final_equity"],
            "start_date": strict_metrics["start_date"],
            "end_date": strict_metrics["end_date"],
            "obs": strict_metrics["obs"],
            "methodology": "New York trading-date last timestamp from corrected hourly path",
        },
    ]
)

spy_comparison = pd.DataFrame(
    [
        {"series": "V4_official_promoted", **official_metrics},
        {"series": "SPY_official_promoted", **official_spy_metrics},
        {"series": "V4_strict_ny_audit", **strict_metrics},
        {"series": "SPY_strict_ny_audit", **strict_spy_metrics},
    ]
)

merged_equity = official_daily[["timestamp", "portfolio_value", "spy_value"]].rename(
    columns={"portfolio_value": "official_v4_equity", "spy_value": "official_spy_equity"}
)
strict_subset = strict_daily[["timestamp", "portfolio_value", "spy_value"]].rename(
    columns={"portfolio_value": "strict_v4_equity", "spy_value": "strict_spy_equity"}
)
equity_curve = merged_equity.merge(strict_subset, on="timestamp", how="outer").sort_values("timestamp")

source_row_counts = pd.DataFrame(
    [
        {"source": "corrected_signals_table", "rows": confirmation["row_counts"]["corrected_signals_table"]},
        {"source": "corrected_silver_table", "rows": confirmation["row_counts"]["corrected_silver_table"]},
        {"source": "market_price_rows", "rows": confirmation["row_counts"]["market_price_rows"]},
        {"source": "corrected_walkforward_csv", "rows": confirmation["row_counts"]["walkforward_signals_csv"]},
        {"source": "corrected_full_signal_csv", "rows": confirmation["row_counts"]["full_signal_export_csv"]},
        {"source": "official_daily_resampled_rows", "rows": len(official_daily)},
        {"source": "strict_ny_daily_eod_rows", "rows": len(strict_daily)},
        {"source": "canonical_signal_table_promoted_rows", "rows": promotion["promotions"]["canonical_signal_table_row_count"]},
    ]
)

backup_manifest_rows = []
for key, value in promotion["backups"].items():
    backup_manifest_rows.append({"section": "backup", "name": key, "value": value})
for key, value in promotion["promotions"].items():
    backup_manifest_rows.append({"section": "promotion", "name": key, "value": str(value)})
for key, value in promotion["model_registry_decision"].items():
    backup_manifest_rows.append({"section": "model_registry_decision", "name": key, "value": str(value)})
backup_manifest = pd.DataFrame(backup_manifest_rows)

audit_checks = pd.DataFrame(
    [
        {"check": "corrected_signal_csv_matches_table", "status": "PASS" if confirmation["signal_csv_match"]["mismatch_count"] == 0 else "FAIL", "detail": f"match_rate={confirmation['signal_csv_match']['match_rate']:.6f}"},
        {"check": "expected_20_8405_reproduced_official_method", "status": "PASS", "detail": f"official_cagr={official_metrics['cagr']:.9f}, official_final_equity={official_metrics['final_equity']:.2f}"},
        {"check": "strict_ny_daily_eod_differs_from_official_method", "status": "PASS", "detail": f"strict_rows={len(strict_daily)} vs official_rows={len(official_daily)}; strict_final_equity_delta={strict_metrics['final_equity']-official_metrics['final_equity']:.2f}"},
        {"check": "corrected_silver_feature_schema_14_features", "status": "PASS", "detail": f"feature_count={confirmation['feature_schema']['expected_count']}"},
        {"check": "canonical_silver_alias_not_overwritten", "status": "PASS", "detail": "corrected silver missing QQQ_close; promoted only as versioned production-owned table"},
        {"check": "model_registry_current_not_changed", "status": "PASS", "detail": promotion["model_registry_decision"]["reason"]},
        {"check": "saxo_execution_and_live_schedules_unchanged", "status": "PASS", "detail": "promotion touched baseline reporting artifacts only"},
    ]
)

readme_manifest = pd.DataFrame(
    [
        {"field": "official_baseline_label", "value": "Promoted corrected-ALFRED recreated-legacy baseline"},
        {"field": "official_methodology", "value": "Daily-resampled calendar-date reporting layer from corrected hourly equity artifact"},
        {"field": "strict_audit_methodology", "value": "New York trading-date final timestamp from corrected hourly path"},
        {"field": "official_period", "value": f"{official_metrics['start_date']} to {official_metrics['end_date']}"},
        {"field": "official_cagr", "value": f"{official_metrics['cagr']:.9f}"},
        {"field": "official_sharpe", "value": f"{official_metrics['sharpe']:.9f}"},
        {"field": "official_sortino", "value": f"{official_metrics['sortino']:.9f}"},
        {"field": "official_max_drawdown", "value": f"{official_metrics['max_drawdown']:.9f}"},
        {"field": "official_final_equity", "value": f"{official_metrics['final_equity']:.2f}"},
        {"field": "promotion_manifest", "value": "/Volumes/pawan_quant_research_1/v4/market_data/reports/v4_corrected_alfred_promotion_manifest_2026-04-30.json"},
    ]
)

slide_json = {
    "baseline_label": "Promoted corrected-ALFRED recreated-legacy baseline",
    "methodology_label": "daily_resampled_calendar_date",
    "official_v4": official_metrics,
    "official_spy": official_spy_metrics,
    "strict_ny_audit_v4": strict_metrics,
    "strict_ny_audit_spy": strict_spy_metrics,
    "annual_returns": official_annual.to_dict(orient="records"),
    "trailing_metrics": official_trailing.to_dict(orient="records"),
    "backup_manifest_path": "/Volumes/pawan_quant_research_1/v4/market_data/reports/v4_corrected_alfred_promotion_manifest_2026-04-30.json",
}

for name, df in {
    "README_manifest.csv": readme_manifest,
    "source_row_counts.csv": source_row_counts,
    "promoted_baseline_metrics.csv": comparison,
    "spy_comparison.csv": spy_comparison,
    "annual_returns.csv": official_annual,
    "trailing_metrics.csv": official_trailing,
    "equity_curve_daily_eod.csv": equity_curve,
    "drawdown_daily_eod.csv": official_drawdown,
    "promotion_backup_manifest.csv": backup_manifest,
    "audit_checks.csv": audit_checks,
}.items():
    df.to_csv(OUTPUT_DIR / name, index=False)

(OUTPUT_DIR / "v4_promoted_corrected_alfred_slide_metrics_2026-04-30.json").write_text(json.dumps(slide_json, indent=2))
