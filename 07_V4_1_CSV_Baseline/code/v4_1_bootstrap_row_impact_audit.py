"""Audit impact of the eight 2016-01-01 V4.1 macro bootstrap rows.

This script rebuilds grids only. It does not train models, write production
Delta tables, update workflows, or mutate any production V4 asset.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


V4_1_CODE_DIR = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/code")
if V4_1_CODE_DIR.exists() and str(V4_1_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(V4_1_CODE_DIR))

from v4_1_build_grid_from_csv import RAW_Z_TARGETS, build_variant_grid
from v4_1_common import FEATURES, INDICATORS, V4_1_ROOT, macro_available_ts, ny_time_to_utc, parse_utc


AUDIT_ROOT = V4_1_ROOT / "bootstrap_row_impact_audit_2026-04-30"
SOURCE_MACRO = (
    V4_1_ROOT
    / "alfred_event_selector_audit_2026-04-30"
    / "outputs"
    / "v4_1_macro_alfred_selected_first_releases.csv"
)
MARKET_INPUT = V4_1_ROOT / "inputs" / "v4_1_market_spy_qqq_1h.csv"
VIX_INPUT = V4_1_ROOT / "inputs" / "v4_1_vix_vix3m_1h_pit.csv"
TIMING_ASSUMPTIONS = V4_1_ROOT / "inputs" / "v4_1_timing_assumptions.csv"
FREEZE_GRID = (
    V4_1_ROOT
    / "forensic_feature_freeze_ladder_2026-04-30"
    / "grids"
    / "D_full_feature_freeze"
    / "v4_1_silver_grid.csv"
)
MAIN_VARIANT = "macro_market_open_or_30m_buffer__vix_next_day_10am"
TARGET_Z_FEATURES = ["GDPC1_z", "UNRATE_z", "UMCSENT_z", "INDPRO_z", "CPIAUCSL_z", "PAYEMS_z"]
KEY_CHECK_FEATURES = [
    "GDPC1_z",
    "UNRATE_z",
    "UMCSENT_z",
    "INDPRO_z",
    "CPI_Event_Active",
    "NFP_Event_Active",
]
EVENT_TIMES = {
    "CPIAUCSL": "08:30",
    "PCE": "08:30",
    "PCEPI": "08:30",
    "PAYEMS": "08:30",
    "UNRATE": "08:30",
    "GDPC1": "08:30",
    "INDPRO": "09:15",
    "UMCSENT": "10:00",
}


def assert_audit_path(path: Path) -> Path:
    root = AUDIT_ROOT.resolve()
    candidate = path.resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Refusing to write outside audit root: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def write_csv(df: pd.DataFrame, path: Path) -> None:
    out = assert_audit_path(path)
    df.to_csv(out, index=False)


def write_json(payload: dict, path: Path) -> None:
    out = assert_audit_path(path)
    with out.open("w", encoding="utf-8") as handle:
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
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    out = df.astype(object).where(pd.notna(df), "")
    cols = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in out.itertuples(index=False):
        vals = [str(v).replace("\n", " ").replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def load_macro_source() -> pd.DataFrame:
    macro = pd.read_csv(SOURCE_MACRO)
    macro = macro[macro["indicator_id"].isin(INDICATORS)].copy()
    if "actual_first_release" not in macro.columns:
        macro["actual_first_release"] = pd.to_numeric(macro["selected_value"], errors="coerce")
    if "observation_date" not in macro.columns:
        macro["observation_date"] = macro["selected_observation_date"]
    if "assumed_release_time_et" not in macro.columns:
        macro["assumed_release_time_et"] = macro["indicator_id"].map(EVENT_TIMES)
    if "release_ts_utc" not in macro.columns:
        macro["release_ts_utc"] = [
            ny_time_to_utc(release_date, event_time)
            for release_date, event_time in zip(macro["release_date"], macro["assumed_release_time_et"])
        ]
    macro["release_ts_utc"] = parse_utc(macro["release_ts_utc"])
    macro["actual_first_release"] = pd.to_numeric(macro["actual_first_release"], errors="coerce")
    return macro.sort_values(["indicator_id", "release_ts_utc"]).reset_index(drop=True)


def bootstrap_mask(macro: pd.DataFrame) -> pd.Series:
    release_dates = pd.to_datetime(macro["release_date"], errors="coerce").dt.date.astype(str)
    return release_dates.eq("2016-01-01")


def build_raw_level_series(macro: pd.DataFrame, market: pd.DataFrame, label: str) -> pd.DataFrame:
    hourly = pd.DataFrame({"timestamp": pd.date_range("2016-01-01 00:00:00", market["timestamp"].max(), freq="h")})
    events = macro.copy()
    events["available_ts_utc"] = [
        macro_available_ts(ts, event_time, MAIN_VARIANT)
        for ts, event_time in zip(events["release_ts_utc"], events["assumed_release_time_et"])
    ]
    frames = []
    for indicator, one in events.groupby("indicator_id", sort=True):
        one = one.sort_values("available_ts_utc")
        merged = pd.merge_asof(
            hourly,
            one[["available_ts_utc", "actual_first_release", "release_date", "observation_date"]].sort_values("available_ts_utc"),
            left_on="timestamp",
            right_on="available_ts_utc",
            direction="backward",
        )
        merged["indicator_id"] = indicator
        merged["variant"] = label
        merged = merged.rename(columns={"actual_first_release": "raw_level"})
        frames.append(merged[["variant", "timestamp", "indicator_id", "raw_level", "available_ts_utc", "release_date", "observation_date"]])
    return pd.concat(frames, ignore_index=True).sort_values(["variant", "timestamp", "indicator_id"])


def compare_grid_to_reference(grid: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = grid[["timestamp", *FEATURES]].merge(
        reference[["timestamp", *FEATURES]],
        on="timestamp",
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    rows = []
    for feature in FEATURES:
        c = pd.to_numeric(merged[f"{feature}_candidate"], errors="coerce")
        r = pd.to_numeric(merged[f"{feature}_reference"], errors="coerce")
        mask = c.notna() & r.notna()
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "variant": label,
                    "feature": feature,
                    "comparable_rows": 0,
                    "match_count": 0,
                    "match_pct": np.nan,
                    "mae": np.nan,
                    "max_abs_diff": np.nan,
                    "first_divergence_ts": None,
                }
            )
            continue
        diff = (c[mask] - r[mask]).abs()
        matches = np.isclose(c[mask], r[mask], rtol=1e-9, atol=1e-9)
        first_divergence_ts = None
        if not bool(matches.all()):
            first_divergence_ts = merged.loc[mask[mask].index[~matches][0], "timestamp"].isoformat()
        rows.append(
            {
                "variant": label,
                "feature": feature,
                "comparable_rows": n,
                "match_count": int(matches.sum()),
                "match_pct": float(matches.sum() / n),
                "mae": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "first_divergence_ts": first_divergence_ts,
            }
        )
    return pd.DataFrame(rows)


def zscore_distribution_delta(grid: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = grid[["timestamp", *TARGET_Z_FEATURES]].merge(
        reference[["timestamp", *TARGET_Z_FEATURES]],
        on="timestamp",
        how="inner",
        suffixes=("_candidate", "_reference"),
    )
    rows = []
    for feature in TARGET_Z_FEATURES:
        c = pd.to_numeric(merged[f"{feature}_candidate"], errors="coerce")
        r = pd.to_numeric(merged[f"{feature}_reference"], errors="coerce")
        mask = c.notna() & r.notna()
        rows.append(
            {
                "variant": label,
                "feature": feature,
                "candidate_mean": float(c[mask].mean()) if mask.any() else np.nan,
                "reference_mean": float(r[mask].mean()) if mask.any() else np.nan,
                "mean_delta": float(c[mask].mean() - r[mask].mean()) if mask.any() else np.nan,
                "candidate_std": float(c[mask].std(ddof=1)) if mask.sum() > 1 else np.nan,
                "reference_std": float(r[mask].std(ddof=1)) if mask.sum() > 1 else np.nan,
                "std_delta": float(c[mask].std(ddof=1) - r[mask].std(ddof=1)) if mask.sum() > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_raw_levels(raw_with: pd.DataFrame, raw_without: pd.DataFrame) -> pd.DataFrame:
    merged = raw_with.merge(
        raw_without,
        on=["timestamp", "indicator_id"],
        how="outer",
        suffixes=("_with_bootstrap", "_without_bootstrap"),
    )
    merged["raw_abs_diff"] = (
        pd.to_numeric(merged["raw_level_with_bootstrap"], errors="coerce")
        - pd.to_numeric(merged["raw_level_without_bootstrap"], errors="coerce")
    ).abs()
    rows = []
    for indicator, one in merged.groupby("indicator_id", sort=True):
        diff_mask = (
            one["raw_level_with_bootstrap"].notna()
            != one["raw_level_without_bootstrap"].notna()
        ) | one["raw_abs_diff"].fillna(0).gt(1e-12)
        rows.append(
            {
                "indicator_id": indicator,
                "rows": int(len(one)),
                "differing_raw_level_rows": int(diff_mask.sum()),
                "first_raw_level_difference_ts": None
                if not diff_mask.any()
                else one.loc[diff_mask, "timestamp"].iloc[0].isoformat(),
                "last_raw_level_difference_ts": None
                if not diff_mask.any()
                else one.loc[diff_mask, "timestamp"].iloc[-1].isoformat(),
                "max_raw_abs_diff": float(one["raw_abs_diff"].max()) if one["raw_abs_diff"].notna().any() else np.nan,
                "with_bootstrap_first_valid_ts": None
                if one["raw_level_with_bootstrap"].first_valid_index() is None
                else one.loc[one["raw_level_with_bootstrap"].first_valid_index(), "timestamp"].isoformat(),
                "without_bootstrap_first_valid_ts": None
                if one["raw_level_without_bootstrap"].first_valid_index() is None
                else one.loc[one["raw_level_without_bootstrap"].first_valid_index(), "timestamp"].isoformat(),
                "corrected_reference_proxy_available": False,
                "corrected_reference_proxy_note": "Corrected feature-freeze grid contains z-score features, not raw macro levels.",
            }
        )
    write_csv(merged, AUDIT_ROOT / "raw_macro_levels_with_vs_without_bootstrap.csv")
    return pd.DataFrame(rows)


def main() -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    macro = load_macro_source()
    boot_mask = bootstrap_mask(macro)
    inventory = macro.loc[boot_mask].copy()
    inventory["bootstrap_reason"] = "release_date_2016_01_01_present_in_v4_1_missing_in_v4_raw_macro_table"
    write_csv(inventory, AUDIT_ROOT / "bootstrap_row_inventory.csv")

    macro_with = macro.copy()
    macro_without = macro.loc[~boot_mask].copy()
    write_csv(macro_with, AUDIT_ROOT / "macro_input_with_bootstrap.csv")
    write_csv(macro_without, AUDIT_ROOT / "macro_input_without_bootstrap.csv")

    market = pd.read_csv(MARKET_INPUT)
    market["timestamp"] = parse_utc(market["timestamp"])
    vix = pd.read_csv(VIX_INPUT)
    vix["timestamp"] = parse_utc(vix["timestamp"])
    vix["available_ts_utc"] = parse_utc(vix["available_ts_utc"])
    assumptions = pd.read_csv(TIMING_ASSUMPTIONS)
    reference = pd.read_csv(FREEZE_GRID)
    reference["timestamp"] = parse_utc(reference["timestamp"])

    grid_with, validation_with = build_variant_grid(V4_1_ROOT, MAIN_VARIANT, market, macro_with, vix, assumptions)
    grid_without, validation_without = build_variant_grid(V4_1_ROOT, MAIN_VARIANT, market, macro_without, vix, assumptions)
    write_csv(grid_with, AUDIT_ROOT / "grid_with_bootstrap.csv")
    write_csv(grid_without, AUDIT_ROOT / "grid_without_bootstrap.csv")

    raw_with = build_raw_level_series(macro_with, market, "with_bootstrap")
    raw_without = build_raw_level_series(macro_without, market, "without_bootstrap")
    write_csv(raw_with, AUDIT_ROOT / "raw_macro_levels_with_bootstrap.csv")
    write_csv(raw_without, AUDIT_ROOT / "raw_macro_levels_without_bootstrap.csv")
    raw_summary = compare_raw_levels(raw_with, raw_without)
    write_csv(raw_summary, AUDIT_ROOT / "raw_macro_level_bootstrap_impact_summary.csv")

    cmp_with = compare_grid_to_reference(grid_with, reference, "with_bootstrap")
    cmp_without = compare_grid_to_reference(grid_without, reference, "without_bootstrap")
    feature_cmp = pd.concat([cmp_with, cmp_without], ignore_index=True)
    wide = feature_cmp.pivot(index="feature", columns="variant", values=["match_pct", "mae", "max_abs_diff"]).reset_index()
    wide.columns = ["_".join([x for x in col if x]).strip("_") for col in wide.columns.to_flat_index()]
    wide["match_pct_delta_without_minus_with"] = wide["match_pct_without_bootstrap"] - wide["match_pct_with_bootstrap"]
    wide["mae_delta_without_minus_with"] = wide["mae_without_bootstrap"] - wide["mae_with_bootstrap"]
    wide["improved_without_bootstrap"] = (wide["match_pct_delta_without_minus_with"] > 0) | (wide["mae_delta_without_minus_with"] < 0)
    feature_out = feature_cmp.merge(
        wide[["feature", "match_pct_delta_without_minus_with", "mae_delta_without_minus_with", "improved_without_bootstrap"]],
        on="feature",
        how="left",
    )
    write_csv(feature_out, AUDIT_ROOT / "bootstrap_impact_feature_comparison.csv")
    write_csv(wide, AUDIT_ROOT / "bootstrap_impact_feature_comparison_wide.csv")

    zdist = pd.concat(
        [
            zscore_distribution_delta(grid_with, reference, "with_bootstrap"),
            zscore_distribution_delta(grid_without, reference, "without_bootstrap"),
        ],
        ignore_index=True,
    )
    write_csv(zdist, AUDIT_ROOT / "bootstrap_impact_zscore_distribution_delta.csv")

    overall = (
        feature_cmp.groupby("variant", as_index=False)
        .agg(match_count=("match_count", "sum"), comparable_rows=("comparable_rows", "sum"))
    )
    overall["overall_feature_match_pct"] = overall["match_count"] / overall["comparable_rows"]
    key_features = wide[wide["feature"].isin(KEY_CHECK_FEATURES)].sort_values("feature")
    improved = wide[wide["improved_without_bootstrap"]].sort_values("mae_delta_without_minus_with")
    worsened = wide[~wide["improved_without_bootstrap"]].sort_values("mae_delta_without_minus_with", ascending=False)
    with_match = float(overall.loc[overall["variant"].eq("with_bootstrap"), "overall_feature_match_pct"].iloc[0])
    without_match = float(overall.loc[overall["variant"].eq("without_bootstrap"), "overall_feature_match_pct"].iloc[0])
    bootstrap_explains = bool(without_match > with_match + 0.01 or any(key_features["match_pct_delta_without_minus_with"] > 0.01))

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_folder": str(AUDIT_ROOT),
        "bootstrap_row_count": int(len(inventory)),
        "feature_match_with_bootstrap": with_match,
        "feature_match_without_bootstrap": without_match,
        "feature_match_delta_without_minus_with": without_match - with_match,
        "bootstrap_rows_explain_feature_mismatch": bootstrap_explains,
        "key_feature_checks": key_features.to_dict(orient="records"),
        "raw_level_impact_summary": raw_summary.to_dict(orient="records"),
        "validation_with_bootstrap_passed": validation_with.get("passed"),
        "validation_without_bootstrap_passed": validation_without.get("passed"),
        "production_change_confirmation": {
            "production_delta_tables_modified": False,
            "production_workflows_modified": False,
            "saxo_execution_modified": False,
            "token_refresh_modified": False,
            "models_trained": False,
        },
    }
    write_json(summary_payload, AUDIT_ROOT / "bootstrap_impact_summary.json")

    lines = [
        "# Bootstrap Row Impact Audit",
        "",
        f"Generated UTC: {summary_payload['generated_at_utc']}",
        "",
        f"Output folder: `{AUDIT_ROOT}`",
        f"Bootstrap rows tested: `{len(inventory)}`",
        "",
        "No production tables were mutated. No forward model training was run.",
        "",
        "## Overall Feature Match",
        "",
        md_table(overall),
        "",
        f"Delta without-minus-with: `{without_match - with_match:.8f}`",
        f"Bootstrap rows explain feature mismatch: `{bootstrap_explains}`",
        "",
        "## Required Feature Checks",
        "",
        md_table(key_features),
        "",
        "## Raw Level Impact",
        "",
        md_table(raw_summary),
        "",
        "## Improved Without Bootstrap",
        "",
        md_table(improved[["feature", "match_pct_delta_without_minus_with", "mae_delta_without_minus_with"]]),
        "",
        "## Worsened Or Unchanged Without Bootstrap",
        "",
        md_table(worsened[["feature", "match_pct_delta_without_minus_with", "mae_delta_without_minus_with"]]),
        "",
        "## Next Smallest Step",
        "",
        "If bootstrap rows do not explain the mismatch, compare V4.1 z-score construction directly against the corrected reference by holding raw forward-filled macro levels fixed and varying only rolling-window endpoints, min-periods, ddof, and bootstrap history length.",
    ]
    assert_audit_path(AUDIT_ROOT / "bootstrap_impact_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(to_jsonable(summary_payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
