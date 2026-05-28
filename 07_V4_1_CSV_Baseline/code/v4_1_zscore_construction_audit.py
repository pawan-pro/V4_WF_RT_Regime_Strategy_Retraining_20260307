"""Audit V4.1 z-score construction against the corrected feature-freeze grid.

This holds non-bootstrap raw forward-filled levels fixed and varies only
z-score math. It does not train models or mutate production V4 assets.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


V4_1_CODE_DIR = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/code")
if V4_1_CODE_DIR.exists() and str(V4_1_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_1_CODE_DIR))

from v4_1_common import FEATURES, parse_utc


V4_1_ROOT = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv")
AUDIT_ROOT = V4_1_ROOT / "zscore_construction_audit_2026-04-30"
BOOTSTRAP_ROOT = V4_1_ROOT / "bootstrap_row_impact_audit_2026-04-30"
GRID_WITHOUT_BOOTSTRAP = BOOTSTRAP_ROOT / "grid_without_bootstrap.csv"
MACRO_RAW_WITHOUT_BOOTSTRAP = BOOTSTRAP_ROOT / "raw_macro_levels_without_bootstrap.csv"
VIX_INPUT = V4_1_ROOT / "inputs" / "v4_1_vix_vix3m_1h_pit.csv"
CORRECTED_GRID = (
    V4_1_ROOT
    / "forensic_feature_freeze_ladder_2026-04-30"
    / "grids"
    / "D_full_feature_freeze"
    / "v4_1_silver_grid.csv"
)
RAW_COLS = ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT", "VIXCLS", "vix3m"]
Z_FEATURES = [f"{x}_z" for x in RAW_COLS]
TARGET_FEATURES = [
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
]
NY = ZoneInfo("America/New_York")


def assert_audit_path(path: Path) -> Path:
    root = AUDIT_ROOT.resolve()
    candidate = path.resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Refusing to write outside audit root: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(assert_audit_path(path), index=False)


def write_json(payload: dict, path: Path) -> None:
    with assert_audit_path(path).open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(float(obj)) else float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "No rows."
    out = df.head(max_rows).copy() if max_rows else df.copy()
    out = out.astype(object).where(pd.notna(out), "")
    cols = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in out.itertuples(index=False):
        vals = [str(v).replace("\n", " ").replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    if max_rows and len(df) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(df)} rows.")
    return "\n".join(lines)


def market_hour_mask(ts: pd.Series) -> pd.Series:
    local = pd.to_datetime(ts, utc=True).dt.tz_convert(NY)
    t = local.dt.time
    return (local.dt.weekday < 5) & (t >= time(9, 30)) & (t <= time(16, 0))


def load_fixed_raw_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid = pd.read_csv(GRID_WITHOUT_BOOTSTRAP)
    grid["timestamp"] = parse_utc(grid["timestamp"])
    reference = pd.read_csv(CORRECTED_GRID)
    reference["timestamp"] = parse_utc(reference["timestamp"])

    macro_long = pd.read_csv(MACRO_RAW_WITHOUT_BOOTSTRAP)
    macro_long["timestamp"] = parse_utc(macro_long["timestamp"])
    macro_wide = (
        macro_long.pivot_table(index="timestamp", columns="indicator_id", values="raw_level", aggfunc="last")
        .reset_index()
        .sort_values("timestamp")
    )

    vix = pd.read_csv(VIX_INPUT)
    vix["timestamp"] = parse_utc(vix["timestamp"])
    vix_wide = (
        vix.pivot_table(index="timestamp", columns="series", values="close", aggfunc="last")
        .rename(columns={"VXVCLS": "vix3m"})
        .reset_index()
        .sort_values("timestamp")
    )

    fixed = grid[["timestamp"]].merge(macro_wide, on="timestamp", how="left")
    fixed = fixed.merge(vix_wide[["timestamp", "VIXCLS", "vix3m"]], on="timestamp", how="left")
    fixed[RAW_COLS] = fixed[RAW_COLS].ffill()
    ref_cols = ["timestamp", *TARGET_FEATURES]
    fixed = fixed.merge(reference[ref_cols], on="timestamp", how="left", suffixes=("", "_corrected_reference"))
    for feature in TARGET_FEATURES:
        fixed = fixed.rename(columns={feature: f"corrected_reference_{feature}"})
    write_csv(fixed, AUDIT_ROOT / "inputs" / "fixed_raw_ffill_levels_for_zscore_audit.csv")
    return fixed, grid, reference


def compute_zscore(
    raw: pd.Series,
    timestamps: pd.Series,
    window: int,
    endpoint: str,
    ddof: int,
    min_periods: int,
    bootstrap_behavior: str,
    backbone: str,
) -> pd.Series:
    raw = pd.to_numeric(raw, errors="coerce")
    if backbone == "all_24_7_hourly_rows":
        stat_raw = raw.shift(1) if endpoint == "current_row_excluded" else raw
        rolling = stat_raw.rolling(window=window, min_periods=min_periods)
        mean = rolling.mean()
        std = rolling.std(ddof=ddof).replace(0, 1.0)
        z = (raw - mean) / std
        count = stat_raw.rolling(window=window, min_periods=1).count()
    elif backbone == "market_hour_only_rows":
        mask = market_hour_mask(timestamps)
        market_raw = raw[mask].reset_index(drop=True)
        stat_raw = market_raw.shift(1) if endpoint == "current_row_excluded" else market_raw
        rolling = stat_raw.rolling(window=window, min_periods=min_periods)
        mean = rolling.mean()
        std = rolling.std(ddof=ddof).replace(0, 1.0)
        z_market = (market_raw - mean) / std
        count_market = stat_raw.rolling(window=window, min_periods=1).count()
        if bootstrap_behavior in {"drop_pre_window_rows", "fill_early_zscores_as_0_until_window_full"}:
            z_market[count_market < window] = np.nan
            if bootstrap_behavior == "fill_early_zscores_as_0_until_window_full":
                z_market = z_market.fillna(0.0)
        z = pd.Series(np.nan, index=raw.index, dtype=float)
        z.loc[mask] = z_market.to_numpy()
        z = z.ffill()
        return z
    else:
        raise ValueError(backbone)

    if bootstrap_behavior == "expanding_until_window_full_then_rolling":
        return z
    if bootstrap_behavior == "fixed_rolling_from_start":
        return z
    if bootstrap_behavior == "drop_pre_window_rows":
        z[count < window] = np.nan
        return z
    if bootstrap_behavior == "fill_early_zscores_as_0_until_window_full":
        z[count < window] = np.nan
        return z.fillna(0.0)
    raise ValueError(bootstrap_behavior)


def compare_z(computed: pd.Series, reference: pd.Series, timestamps: pd.Series) -> dict:
    c = pd.to_numeric(computed, errors="coerce")
    r = pd.to_numeric(reference, errors="coerce")
    mask = c.notna() & r.notna()
    n = int(mask.sum())
    if n == 0:
        return {
            "comparable_rows": 0,
            "mae": np.nan,
            "median_abs_error": np.nan,
            "max_abs_error": np.nan,
            "correlation": np.nan,
            "near_match_rate": np.nan,
            "first_divergence_timestamp": None,
            "large_error_gt_0_01": 0,
            "large_error_gt_0_05": 0,
            "large_error_gt_0_10": 0,
        }
    diff = (c[mask] - r[mask]).abs()
    near = diff <= 1e-9
    first = None if bool(near.all()) else timestamps.loc[mask[mask].index[~near][0]].isoformat()
    corr = c[mask].corr(r[mask]) if n > 1 and c[mask].std(ddof=1) != 0 and r[mask].std(ddof=1) != 0 else np.nan
    return {
        "comparable_rows": n,
        "mae": float(diff.mean()),
        "median_abs_error": float(diff.median()),
        "max_abs_error": float(diff.max()),
        "correlation": float(corr) if pd.notna(corr) else np.nan,
        "near_match_rate": float(near.sum() / n),
        "first_divergence_timestamp": first,
        "large_error_gt_0_01": int((diff > 0.01).sum()),
        "large_error_gt_0_05": int((diff > 0.05).sum()),
        "large_error_gt_0_10": int((diff > 0.10).sum()),
    }


def simplicity_rank(row: pd.Series) -> tuple:
    window_order = {8760: 0, 8761: 1, 8759: 2, 8784: 3}
    min_order = {24: 0, 1: 1, 168: 2, 8760: 3}
    bootstrap_order = {
        "expanding_until_window_full_then_rolling": 0,
        "fixed_rolling_from_start": 1,
        "drop_pre_window_rows": 2,
        "fill_early_zscores_as_0_until_window_full": 3,
    }
    return (
        0 if row["backbone"] == "all_24_7_hourly_rows" else 1,
        0 if row["endpoint"] == "current_row_included" else 1,
        0 if int(row["ddof"]) == 1 else 1,
        min_order.get(int(row["min_periods"]), 99),
        window_order.get(int(row["window_length"]), 99),
        bootstrap_order.get(row["bootstrap_behavior"], 99),
    )


def run_variant_matrix(fixed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    windows = [8759, 8760, 8761, 8784]
    endpoints = ["current_row_included", "current_row_excluded"]
    ddofs = [1, 0]
    min_periods_values = [1, 24, 168, 8760]
    bootstrap_behaviors = [
        "expanding_until_window_full_then_rolling",
        "fixed_rolling_from_start",
        "drop_pre_window_rows",
        "fill_early_zscores_as_0_until_window_full",
    ]
    backbones = ["all_24_7_hourly_rows", "market_hour_only_rows"]

    rows = []
    best_series: dict[str, pd.Series] = {}
    best_rows = []
    for raw_col, feature in zip(RAW_COLS, TARGET_FEATURES):
        reference_col = f"corrected_reference_{feature}"
        feature_rows = []
        feature_series: list[tuple[dict, pd.Series]] = []
        for window in windows:
            for endpoint in endpoints:
                for ddof in ddofs:
                    for min_periods in min_periods_values:
                        for bootstrap_behavior in bootstrap_behaviors:
                            for backbone in backbones:
                                if min_periods > window:
                                    row = {
                                        "feature": feature,
                                        "raw_column": raw_col,
                                        "window_length": window,
                                        "endpoint": endpoint,
                                        "ddof": ddof,
                                        "std_type": "sample" if ddof == 1 else "population",
                                        "min_periods": min_periods,
                                        "bootstrap_behavior": bootstrap_behavior,
                                        "backbone": backbone,
                                        "invalid_variant_reason": "min_periods_greater_than_window",
                                        "comparable_rows": 0,
                                        "mae": np.nan,
                                        "median_abs_error": np.nan,
                                        "max_abs_error": np.nan,
                                        "correlation": np.nan,
                                        "near_match_rate": np.nan,
                                        "first_divergence_timestamp": None,
                                        "large_error_gt_0_01": 0,
                                        "large_error_gt_0_05": 0,
                                        "large_error_gt_0_10": 0,
                                    }
                                    feature_rows.append(row)
                                    continue
                                z = compute_zscore(
                                    fixed[raw_col],
                                    fixed["timestamp"],
                                    window,
                                    endpoint,
                                    ddof,
                                    min_periods,
                                    bootstrap_behavior,
                                    backbone,
                                )
                                metrics = compare_z(z, fixed[reference_col], fixed["timestamp"])
                                row = {
                                    "feature": feature,
                                    "raw_column": raw_col,
                                    "window_length": window,
                                    "endpoint": endpoint,
                                    "ddof": ddof,
                                    "std_type": "sample" if ddof == 1 else "population",
                                    "min_periods": min_periods,
                                    "bootstrap_behavior": bootstrap_behavior,
                                    "backbone": backbone,
                                    "invalid_variant_reason": None,
                                    **metrics,
                                }
                                feature_rows.append(row)
                                feature_series.append((row, z))
        rows.extend(feature_rows)
        ranked = pd.DataFrame(feature_rows).copy()
        ranked = ranked[ranked["invalid_variant_reason"].isna()].copy()
        ranked["simplicity_rank"] = ranked.apply(simplicity_rank, axis=1)
        ranked = ranked.sort_values(
            ["mae", "correlation", "max_abs_error", "simplicity_rank"],
            ascending=[True, False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        best = ranked.iloc[0].to_dict()
        best_rows.append(best)
        best_key = (
            best["window_length"],
            best["endpoint"],
            best["ddof"],
            best["min_periods"],
            best["bootstrap_behavior"],
            best["backbone"],
        )
        for row, z in feature_series:
            key = (
                row["window_length"],
                row["endpoint"],
                row["ddof"],
                row["min_periods"],
                row["bootstrap_behavior"],
                row["backbone"],
            )
            if key == best_key:
                best_series[feature] = z
                break

    matrix = pd.DataFrame(rows)
    best_df = pd.DataFrame(best_rows)
    write_csv(matrix, AUDIT_ROOT / "forensics" / "zscore_variant_matrix.csv")
    write_csv(best_df, AUDIT_ROOT / "forensics" / "best_zscore_rule_by_feature.csv")
    return matrix, best_df, best_series


def compare_grid(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    merged = candidate[["timestamp", *FEATURES]].merge(
        reference[["timestamp", *FEATURES]], on="timestamp", how="inner", suffixes=("_candidate", "_reference")
    )
    rows = []
    for feature in FEATURES:
        c = pd.to_numeric(merged[f"{feature}_candidate"], errors="coerce")
        r = pd.to_numeric(merged[f"{feature}_reference"], errors="coerce")
        mask = c.notna() & r.notna()
        if not mask.any():
            rows.append(
                {
                    "feature": feature,
                    "comparable_rows": 0,
                    "match_count": 0,
                    "match_pct": np.nan,
                    "mae": np.nan,
                    "max_abs_diff": np.nan,
                    "first_divergence_timestamp": None,
                }
            )
            continue
        diff = (c[mask] - r[mask]).abs()
        match = diff <= 1e-9
        first = None if bool(match.all()) else merged.loc[mask[mask].index[~match][0], "timestamp"].isoformat()
        rows.append(
            {
                "feature": feature,
                "comparable_rows": int(mask.sum()),
                "match_count": int(match.sum()),
                "match_pct": float(match.sum() / mask.sum()),
                "mae": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "first_divergence_timestamp": first,
            }
        )
    return pd.DataFrame(rows)


def build_patch_grid(grid: pd.DataFrame, reference: pd.DataFrame, best_series: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patched = grid.copy()
    before = compare_grid(grid, reference)
    for feature, z in best_series.items():
        patched[feature] = z.to_numpy()
    patched["VIX_Inversion_Z"] = patched["VIXCLS_z"] - patched["vix3m_z"]
    write_csv(patched, AUDIT_ROOT / "grids" / "v4_1_silver_grid_best_zscore_patch.csv")
    after = compare_grid(patched, reference)
    combined = pd.concat(
        [
            before.assign(grid_variant="before_nonbootstrap"),
            after.assign(grid_variant="after_best_zscore_patch"),
        ],
        ignore_index=True,
    )
    write_csv(combined, AUDIT_ROOT / "forensics" / "best_zscore_patch_feature_match.csv")
    return patched, before, after


def overall_match(df: pd.DataFrame) -> float:
    return float(df["match_count"].sum() / df["comparable_rows"].sum())


def main() -> None:
    (AUDIT_ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "forensics").mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "grids").mkdir(parents=True, exist_ok=True)

    fixed, grid, reference = load_fixed_raw_frame()
    matrix, best_df, best_series = run_variant_matrix(fixed)
    patched, before, after = build_patch_grid(grid, reference, best_series)

    before_match = overall_match(before)
    after_match = overall_match(after)
    macro_before = before[before["feature"].isin(TARGET_FEATURES)]
    macro_after = after[after["feature"].isin(TARGET_FEATURES)]
    macro_before_match = overall_match(macro_before)
    macro_after_match = overall_match(macro_after)
    vix_best = best_df[best_df["feature"].isin(["VIXCLS_z", "vix3m_z"])].copy()
    best_rule_counts = (
        best_df.groupby(["window_length", "endpoint", "std_type", "min_periods", "bootstrap_behavior", "backbone"], as_index=False)
        .agg(feature_count=("feature", "count"), features=("feature", lambda s: ",".join(s)))
        .sort_values("feature_count", ascending=False)
    )
    one_global_rule = bool(best_rule_counts["feature_count"].max() == len(TARGET_FEATURES))
    material_improvement = after_match > before_match + 0.05
    vix_explained = bool(vix_best["mae"].max() < 0.01 and vix_best["near_match_rate"].min() > 0.95)
    next_issue = (
        "event flags" if material_improvement and not after[after["feature"].isin(["CPI_Event_Active", "NFP_Event_Active"])]["match_pct"].min() > 0.95
        else "raw VIX/source lineage" if not vix_explained
        else "forward-fill/bootstrap history"
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_folder": str(AUDIT_ROOT),
        "fixed_raw_level_audit_csv": str(AUDIT_ROOT / "inputs" / "fixed_raw_ffill_levels_for_zscore_audit.csv"),
        "zscore_matrix_path": str(AUDIT_ROOT / "forensics" / "zscore_variant_matrix.csv"),
        "best_rule_table_path": str(AUDIT_ROOT / "forensics" / "best_zscore_rule_by_feature.csv"),
        "patched_grid_path": str(AUDIT_ROOT / "grids" / "v4_1_silver_grid_best_zscore_patch.csv"),
        "feature_match_before_nonbootstrap": before_match,
        "feature_match_after_best_zscore_patch": after_match,
        "feature_match_delta": after_match - before_match,
        "zscore_feature_match_before": macro_before_match,
        "zscore_feature_match_after": macro_after_match,
        "zscore_feature_match_delta": macro_after_match - macro_before_match,
        "one_global_rule": one_global_rule,
        "best_rule_counts": best_rule_counts.to_dict(orient="records"),
        "zscore_construction_explains_grid_mismatch": material_improvement,
        "vix_vix3m_explained_by_zscore_construction": vix_explained,
        "next_issue": next_issue,
        "production_change_confirmation": {
            "production_delta_tables_modified": False,
            "production_workflows_modified": False,
            "saxo_execution_modified": False,
            "token_refresh_modified": False,
            "rt_notebooks_modified": False,
            "dashboard_notebooks_modified": False,
            "model_registry_modified": False,
            "models_trained": False,
        },
    }
    write_json(summary, AUDIT_ROOT / "forensics" / "zscore_construction_audit_summary.json")

    key_cols = [
        "feature",
        "window_length",
        "endpoint",
        "std_type",
        "min_periods",
        "bootstrap_behavior",
        "backbone",
        "mae",
        "correlation",
        "max_abs_error",
        "near_match_rate",
        "first_divergence_timestamp",
    ]
    patch_wide = before[["feature", "match_pct", "mae", "max_abs_diff"]].merge(
        after[["feature", "match_pct", "mae", "max_abs_diff"]],
        on="feature",
        suffixes=("_before", "_after"),
    )
    patch_wide["match_pct_delta"] = patch_wide["match_pct_after"] - patch_wide["match_pct_before"]
    patch_wide["mae_delta"] = patch_wide["mae_after"] - patch_wide["mae_before"]
    lines = [
        "# Z-Score Construction Audit",
        "",
        f"Generated UTC: {summary['generated_at_utc']}",
        "",
        f"Output folder: `{AUDIT_ROOT}`",
        "",
        "No raw release values, release timestamps, production tables, models, Saxo assets, workflows, or schedules were changed.",
        "",
        "## Overall Feature Match",
        "",
        md_table(
            pd.DataFrame(
                [
                    {"grid": "before_nonbootstrap", "feature_match": before_match},
                    {"grid": "after_best_zscore_patch", "feature_match": after_match},
                    {"grid": "delta", "feature_match": after_match - before_match},
                ]
            )
        ),
        "",
        "## Z-Score Feature Match",
        "",
        md_table(
            pd.DataFrame(
                [
                    {"grid": "before_nonbootstrap", "zscore_feature_match": macro_before_match},
                    {"grid": "after_best_zscore_patch", "zscore_feature_match": macro_after_match},
                    {"grid": "delta", "zscore_feature_match": macro_after_match - macro_before_match},
                ]
            )
        ),
        "",
        f"One global rule: `{one_global_rule}`",
        f"Z-score construction explains grid mismatch: `{material_improvement}`",
        f"VIX/VIX3M explained by z-score construction: `{vix_explained}`",
        f"Next issue: `{next_issue}`",
        "",
        "## Best Rule By Feature",
        "",
        md_table(best_df[key_cols].sort_values("feature")),
        "",
        "## Patch Feature Match",
        "",
        md_table(patch_wide.sort_values("mae_after", ascending=False)),
        "",
        "## Rule Count",
        "",
        md_table(best_rule_counts),
    ]
    assert_audit_path(AUDIT_ROOT / "forensics" / "zscore_construction_audit_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
