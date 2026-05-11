# Databricks notebook source
"""MUB-27 V4.2 leverage sensitivity.

Research-only mathematical leverage sensitivity on existing isolated strict-PIT
daily return outputs. This script writes only under:

`/Volumes/pawan_quant_research_1/v4/market_data/v4_2_leverage_sensitivity_2026-05-06/`

It does not write production Delta tables, mutate Databricks jobs/workflows,
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
BASE = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data")
OUTPUT_ROOT = BASE / "v4_2_leverage_sensitivity_2026-05-06"
CHARTS_DIR = OUTPUT_ROOT / "charts"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"

STRICT_ROOT = BASE / "v4_2_strict_pit_research_2026-05-06"
BILSTM_ROOT = BASE / "v4_2_bilstm_strict_pit_2026-05-06"
TFT_ROOT = BASE / "v4_2_tft_strict_pit_2026-05-06"
WINDOW_ROOT = BASE / "v4_2_tft_window_defensive_sleeve_2026-05-06"

INITIAL_CAPITAL = 5_000_000.0
CAGR_YEARS = 5.0
TRADING_DAYS = 252
LEVERAGE_GRID = [1.0, 2.0, 3.0, 5.0]
FINANCING_GRID = [0.00, 0.03, 0.05, 0.07, 0.10]

UNLEVERED_SPY_CAGR = 0.127546
B_C2_CAGR = 0.208405


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    path: Path
    pit_status: str
    return_column: str = "strategy_daily_return"
    filter_variant: str | None = None


VARIANTS = [
    VariantSpec(
        "A_raw_label_walkforward",
        STRICT_ROOT / "performance" / "A_raw_label_walkforward_daily_returns.csv",
        "strict_pit",
    ),
    VariantSpec(
        "C_strict_rolling_teacher_student",
        STRICT_ROOT / "performance" / "C_strict_rolling_teacher_student_daily_returns.csv",
        "strict_pit",
    ),
    VariantSpec(
        "D_pca_strict_xgboost",
        STRICT_ROOT / "performance" / "D_pca_strict_xgboost_daily_returns.csv",
        "strict_pit",
    ),
    VariantSpec(
        "BiLSTM_raw_24h",
        BILSTM_ROOT / "performance" / "BiLSTM_raw_24h_daily_returns.csv",
        "strict_pit",
    ),
    VariantSpec(
        "TFT_attention_raw_72h__CASH",
        WINDOW_ROOT / "performance" / "daily_returns_by_sleeve.csv",
        "strict_pit",
        filter_variant="TFT_attention_raw_72h__CASH",
    ),
    VariantSpec(
        "TFT_attention_raw_72h__BIL",
        WINDOW_ROOT / "performance" / "daily_returns_by_sleeve.csv",
        "strict_pit",
        filter_variant="TFT_attention_raw_72h__BIL",
    ),
    VariantSpec(
        "E_spy_benchmark",
        STRICT_ROOT / "performance" / "E_spy_benchmark_daily_returns.csv",
        "benchmark",
    ),
    VariantSpec(
        "B_reference_teacher_smoothed",
        STRICT_ROOT / "performance" / "B_reference_teacher_smoothed_daily_returns.csv",
        "non_pit_reference_only",
    ),
]

COMPARISON_FILES = {
    "strict_pit_variant_comparison": STRICT_ROOT / "forensics" / "variant_comparison.csv",
    "bilstm_comparison": BILSTM_ROOT / "forensics" / "comparison_vs_A_B_C_D_SPY.csv",
    "tft_comparison": TFT_ROOT / "forensics" / "comparison_vs_A_B_C_D_SPY_BiLSTM.csv",
    "window_defensive_full_comparison": WINDOW_ROOT / "forensics" / "full_variant_comparison.csv",
}


def ensure_python_deps() -> None:
    try:
        import matplotlib  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])


ensure_python_deps()
import matplotlib.pyplot as plt


def ensure_dirs() -> None:
    for path in [OUTPUT_ROOT, CHARTS_DIR, MANIFESTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def assert_output_path(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside isolated research root: {path}")


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


def normalize_date_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "report_date" in out.columns:
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce").dt.date.astype(str)
    elif "timestamp" in out.columns:
        out["report_date"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.date.astype(str)
    else:
        raise ValueError("Expected report_date or timestamp column")
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out.dropna(subset=["report_date"]).sort_values("report_date").reset_index(drop=True)


def load_variant_returns(spec: VariantSpec) -> pd.DataFrame:
    df = normalize_date_frame(pd.read_csv(spec.path))
    if spec.filter_variant is not None:
        if "variant" not in df.columns:
            raise ValueError(f"{spec.path} missing variant column for {spec.filter_variant}")
        df = df[df["variant"].astype(str).eq(spec.filter_variant)].copy()
    if spec.return_column not in df.columns:
        raise ValueError(f"{spec.path} missing return column {spec.return_column}")
    out = df[["report_date", spec.return_column]].rename(columns={spec.return_column: "base_daily_return"}).copy()
    out["base_daily_return"] = pd.to_numeric(out["base_daily_return"], errors="coerce").fillna(0.0)
    out = out.drop_duplicates("report_date").sort_values("report_date").reset_index(drop=True)
    out["variant"] = spec.variant
    out["pit_status"] = spec.pit_status
    out["source_path"] = str(spec.path)
    return out


def liquidation_flag(max_dd: float, breached_zero: bool) -> str:
    if breached_zero:
        return "failed"
    if max_dd <= -0.80:
        return "severe"
    if max_dd <= -0.60:
        return "high"
    return "normal"


def compute_metric_row(
    variant: str,
    pit_status: str,
    leverage: float,
    annual_financing_rate: float,
    base_returns: pd.Series,
    dates: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = base_returns.astype(float).to_numpy()
    daily_financing_rate = annual_financing_rate / TRADING_DAYS
    levered_raw = leverage * raw - max(leverage - 1.0, 0.0) * daily_financing_rate
    factors = 1.0 + levered_raw
    equity_values = []
    effective_returns = []
    equity = INITIAL_CAPITAL
    breached_zero = False
    for ret, factor in zip(levered_raw, factors):
        if breached_zero:
            effective_returns.append(0.0)
            equity_values.append(0.0)
            continue
        if factor <= 0.0:
            breached_zero = True
            effective_returns.append(-1.0)
            equity = 0.0
            equity_values.append(0.0)
            continue
        effective_returns.append(float(ret))
        equity *= float(factor)
        equity_values.append(equity)

    returns = pd.Series(effective_returns, dtype=float)
    equity_series = pd.Series(equity_values, dtype=float)
    risk_returns = returns.iloc[1:] if len(returns) > 1 else returns
    roll_max = equity_series.cummax().replace(0.0, np.nan)
    drawdown = equity_series / roll_max - 1.0
    drawdown = drawdown.fillna(-1.0 if breached_zero else 0.0)
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    final_equity = float(equity_series.iloc[-1]) if len(equity_series) else INITIAL_CAPITAL
    cagr = -1.0 if breached_zero else float((final_equity / INITIAL_CAPITAL) ** (1.0 / CAGR_YEARS) - 1.0)
    vol = float(risk_returns.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(risk_returns) > 1 else np.nan
    sharpe = float(risk_returns.mean() / risk_returns.std(ddof=1) * math.sqrt(TRADING_DAYS)) if risk_returns.std(ddof=1) else np.nan
    downside = risk_returns[risk_returns < 0].std(ddof=1)
    sortino = float(risk_returns.mean() / downside * math.sqrt(TRADING_DAYS)) if downside and not np.isnan(downside) else np.nan
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan
    flag = liquidation_flag(max_dd, breached_zero)
    row = {
        "variant": variant,
        "pit_status": pit_status,
        "leverage": leverage,
        "annual_financing_rate": annual_financing_rate,
        "daily_financing_rate": daily_financing_rate,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max DD": max_dd,
        "Final Equity": final_equity,
        "worst_daily_return": float(returns.min()) if len(returns) else np.nan,
        "best_daily_return": float(returns.max()) if len(returns) else np.nan,
        "annualized_volatility": vol,
        "Calmar": calmar,
        "min_equity": float(equity_series.min()) if len(equity_series) else np.nan,
        "breached_zero": breached_zero,
        "max_one_day_loss": float(returns.min()) if len(returns) else np.nan,
        "liquidation_risk_flag": flag,
        "gap_vs_unlevered_SPY_CAGR": cagr - UNLEVERED_SPY_CAGR,
        "gap_vs_B_C2_CAGR": cagr - B_C2_CAGR,
    }
    curve = pd.DataFrame(
        {
            "report_date": dates.astype(str).to_numpy(),
            "variant": variant,
            "pit_status": pit_status,
            "leverage": leverage,
            "annual_financing_rate": annual_financing_rate,
            "base_daily_return": raw,
            "levered_daily_return": returns.to_numpy(),
            "levered_equity": equity_series.to_numpy(),
            "drawdown": drawdown.to_numpy(),
        }
    )
    return row, curve


def build_leverage_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    curve_frames = []
    for spec in VARIANTS:
        returns = load_variant_returns(spec)
        for leverage in LEVERAGE_GRID:
            for financing in FINANCING_GRID:
                row, curve = compute_metric_row(
                    spec.variant,
                    spec.pit_status,
                    leverage,
                    financing,
                    returns["base_daily_return"],
                    returns["report_date"],
                )
                summary_rows.append(row)
                curve_frames.append(curve)
    return pd.DataFrame(summary_rows), pd.concat(curve_frames, ignore_index=True)


def make_best_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = summary[~summary["pit_status"].eq("non_pit_reference_only")].copy()
    best_by_variant = (
        eligible.sort_values(["variant", "CAGR", "Sharpe"], ascending=[True, False, False])
        .groupby("variant", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_leverage = (
        eligible.sort_values(["leverage", "annual_financing_rate", "CAGR", "Sharpe"], ascending=[True, True, False, False])
        .groupby(["leverage", "annual_financing_rate"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    risk_flags = summary[
        [
            "variant",
            "pit_status",
            "leverage",
            "annual_financing_rate",
            "Max DD",
            "min_equity",
            "breached_zero",
            "liquidation_risk_flag",
            "worst_daily_return",
            "gap_vs_unlevered_SPY_CAGR",
            "gap_vs_B_C2_CAGR",
        ]
    ].copy()
    return best_by_variant, best_by_leverage, risk_flags


def plot_tft_bil_curves(curves: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    subset = curves[
        curves["variant"].eq("TFT_attention_raw_72h__BIL")
        & curves["annual_financing_rate"].isin([0.0, 0.05, 0.10])
        & curves["leverage"].isin(LEVERAGE_GRID)
    ].copy()
    for (lev, fin), grp in subset.groupby(["leverage", "annual_financing_rate"]):
        label = f"{lev:g}x, {fin:.0%} fin"
        ax.plot(pd.to_datetime(grp["report_date"]), grp["levered_equity"] / 1_000_000, linewidth=1.1, label=label)
    ax.set_title("TFT 72h BIL Leverage Curves")
    ax.set_ylabel("Equity ($M)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    path = CHARTS_DIR / "tft_72h_bil_leverage_curves.png"
    assert_output_path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    strict = summary[summary["pit_status"].eq("strict_pit")].copy()
    scatter = ax.scatter(strict["Max DD"], strict["CAGR"], c=strict["leverage"], cmap="viridis", alpha=0.78)
    ax.axhline(UNLEVERED_SPY_CAGR, color="black", linestyle="--", linewidth=1.0, label="Unlevered SPY CAGR")
    ax.axhline(B_C2_CAGR, color="tab:red", linestyle=":", linewidth=1.0, label="B/C2 CAGR")
    ax.set_title("Strict-PIT Leverage CAGR vs Drawdown")
    ax.set_xlabel("Max Drawdown")
    ax.set_ylabel("CAGR")
    ax.grid(True, alpha=0.25)
    ax.legend()
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Leverage")
    path = CHARTS_DIR / "leverage_cagr_vs_drawdown_scatter.png"
    assert_output_path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame) -> None:
    subset = summary[summary["variant"].eq("TFT_attention_raw_72h__BIL")].copy()
    pivot = subset.pivot(index="annual_financing_rate", columns="leverage", values="CAGR").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}x" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.0%}" for x in pivot.index])
    ax.set_title("TFT 72h BIL CAGR Heatmap")
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Annual Financing")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1%}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="CAGR")
    path = CHARTS_DIR / "leverage_grid_heatmap_cagr.png"
    assert_output_path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_readme(summary: pd.DataFrame, best_by_variant: pd.DataFrame) -> None:
    tft_bil = summary[summary["variant"].eq("TFT_attention_raw_72h__BIL")]
    best_tft = best_by_variant[best_by_variant["variant"].eq("TFT_attention_raw_72h__BIL")].iloc[0].to_dict()
    strict_beats_spy = summary[(summary["pit_status"].eq("strict_pit")) & (summary["CAGR"] > UNLEVERED_SPY_CAGR)]
    lines = [
        "# MUB-27 V4.2 Leverage Sensitivity",
        "",
        "Research-only mathematical leverage sensitivity. This is not a production trading recommendation.",
        "",
        "## Governance",
        "",
        "- Inputs are existing isolated MUB-27 research daily return files.",
        "- Existing switching costs embedded in base daily returns are preserved; transaction costs are not double-counted.",
        "- No production V4 Delta tables, Saxo execution, token refresh, RT/dashboard notebooks, live schedules, model registry entries, or trading logic were changed.",
        "- 5x leverage is reported only as a mathematical sensitivity. It is not presented as investable.",
        "- CFD leverage requires a separate swap/financing model before any implementation discussion.",
        "",
        "## Key Findings",
        "",
        f"- Best TFT 72h BIL leverage row: {best_tft}",
        f"- Strict-PIT rows beating unlevered SPY CAGR: {len(strict_beats_spy)}",
        f"- TFT 72h BIL 5x rows: {tft_bil[tft_bil['leverage'].eq(5.0)].to_dict(orient='records')}",
        "",
    ]
    write_text("\n".join(lines), OUTPUT_ROOT / "README.md")


def write_manifests(summary: pd.DataFrame, reconciliation: dict[str, Any]) -> None:
    input_hashes = {
        "variant_daily_return_files": {
            spec.variant: {
                "path": spec.path,
                "sha256": file_sha256(spec.path),
                "filter_variant": spec.filter_variant,
            }
            for spec in VARIANTS
        },
        "comparison_files": {
            name: {"path": path, "sha256": file_sha256(path)} for name, path in COMPARISON_FILES.items()
        },
    }
    write_json(input_hashes, MANIFESTS_DIR / "input_hashes.json")
    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
        "output_root": OUTPUT_ROOT,
        "leverage_grid": LEVERAGE_GRID,
        "financing_grid": FINANCING_GRID,
        "methodology": {
            "formula": "levered_daily_return = L * strategy_daily_return - max(L - 1, 0) * annual_financing_rate / 252",
            "compounding": "daily",
            "base_transaction_costs": "preserved from prior daily returns; not double-counted",
            "cagr_years": CAGR_YEARS,
            "sharpe_sortino": "daily returns excluding first zero row, 252 annualization, risk-free 0",
        },
        "reconciliation": reconciliation,
        "production_delta_writes": False,
        "workflow_or_schedule_changes": False,
        "saxo_or_token_changes": False,
        "rt_or_dashboard_changes": False,
        "model_registry_writes": False,
        "trading_logic_changes": False,
        "rows_written": int(len(summary)),
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")


def main() -> None:
    ensure_dirs()
    summary, curves = build_leverage_outputs()
    best_by_variant, best_by_leverage, risk_flags = make_best_tables(summary)

    tft_1x = summary[
        summary["variant"].eq("TFT_attention_raw_72h__BIL")
        & summary["leverage"].eq(1.0)
        & summary["annual_financing_rate"].eq(0.0)
    ].iloc[0]
    reconciliation = {
        "TFT_attention_raw_72h__BIL_1x_0pct": {
            "CAGR": tft_1x["CAGR"],
            "Sharpe": tft_1x["Sharpe"],
            "Max DD": tft_1x["Max DD"],
            "expected_CAGR_approx": 0.081612,
            "expected_Sharpe_approx": 0.6056,
            "expected_Max_DD_approx": -0.244077,
            "reconciles": bool(
                abs(tft_1x["CAGR"] - 0.081612) < 0.0001
                and abs(tft_1x["Sharpe"] - 0.6056) < 0.0002
                and abs(tft_1x["Max DD"] - -0.244077) < 0.0001
            ),
        }
    }

    write_csv(summary.sort_values(["variant", "leverage", "annual_financing_rate"]), OUTPUT_ROOT / "leverage_summary.csv")
    write_csv(curves.sort_values(["variant", "leverage", "annual_financing_rate", "report_date"]), OUTPUT_ROOT / "leverage_curves.csv")
    write_csv(risk_flags.sort_values(["variant", "leverage", "annual_financing_rate"]), OUTPUT_ROOT / "leverage_risk_flags.csv")
    write_csv(best_by_variant, OUTPUT_ROOT / "best_by_variant.csv")
    write_csv(best_by_leverage, OUTPUT_ROOT / "best_by_leverage.csv")

    plot_tft_bil_curves(curves)
    plot_scatter(summary)
    plot_heatmap(summary)
    write_readme(summary, best_by_variant)
    write_manifests(summary, reconciliation)

    print(summary.to_string(index=False))
    print(json.dumps(jsonable({"reconciliation": reconciliation}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
