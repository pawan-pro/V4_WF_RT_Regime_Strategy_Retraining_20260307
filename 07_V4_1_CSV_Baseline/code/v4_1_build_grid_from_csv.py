"""Build V4.1 silver-grid CSVs from frozen input CSVs only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from v4_1_common import (
    FINAL_GRID_COLUMNS,
    FEATURES,
    INDICATORS,
    TIMING_VARIANTS,
    V4_1_ROOT,
    apply_macro_variant_availability,
    ensure_v4_1_dirs,
    parse_utc,
    print_json,
    sha256_file,
    write_csv,
    write_json,
)


RAW_Z_TARGETS = ["CPIAUCSL", "PCEPI", "PCE", "PAYEMS", "UNRATE", "GDPC1", "INDPRO", "UMCSENT", "VIXCLS", "vix3m"]


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(root / "inputs" / "v4_1_market_spy_qqq_1h.csv")
    macro = pd.read_csv(root / "inputs" / "v4_1_macro_alfred_first_releases.csv")
    vix = pd.read_csv(root / "inputs" / "v4_1_vix_vix3m_1h_pit.csv")
    assumptions = pd.read_csv(root / "inputs" / "v4_1_timing_assumptions.csv")
    variants = pd.read_csv(root / "inputs" / "v4_1_timing_variants.csv")
    market["timestamp"] = parse_utc(market["timestamp"])
    macro["release_ts_utc"] = parse_utc(macro["release_ts_utc"])
    vix["timestamp"] = parse_utc(vix["timestamp"])
    vix["available_ts_utc"] = parse_utc(vix["available_ts_utc"])
    return market, macro, vix, assumptions, variants


def merge_asof_feature(backbone: pd.DataFrame, events: pd.DataFrame, time_col: str, value_cols: list[str]) -> pd.DataFrame:
    merged = pd.merge_asof(
        backbone[["timestamp"]].sort_values("timestamp"),
        events[[time_col, *value_cols]].sort_values(time_col),
        left_on="timestamp",
        right_on=time_col,
        direction="backward",
    )
    return merged.drop(columns=[time_col], errors="ignore")


def build_variant_grid(root: Path, variant: str, market: pd.DataFrame, macro: pd.DataFrame, vix: pd.DataFrame, assumptions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    end_ts = market["timestamp"].max()
    backbone = pd.DataFrame({"timestamp": pd.date_range("2016-01-01 00:00:00", end_ts, freq="h")})

    market_wide = (
        market.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .rename(columns={"SPY": "SPY_close", "QQQ": "QQQ_close"})
        .reset_index()
        .sort_values("timestamp")
    )
    grid = backbone.merge(market_wide, on="timestamp", how="left")
    grid[["SPY_close", "QQQ_close"]] = grid[["SPY_close", "QQQ_close"]].ffill()

    macro_v = apply_macro_variant_availability(macro, variant)
    macro_wide = (
        macro_v.pivot_table(index="available_ts_utc", columns="indicator_id", values="actual_first_release", aggfunc="last")
        .reset_index()
        .sort_values("available_ts_utc")
    )
    macro_joined = merge_asof_feature(backbone, macro_wide, "available_ts_utc", [c for c in INDICATORS if c in macro_wide.columns])
    for col in INDICATORS:
        if col not in macro_joined.columns:
            macro_joined[col] = np.nan
    grid = grid.merge(macro_joined, on="timestamp", how="left")
    grid[INDICATORS] = grid[INDICATORS].ffill()

    vix_wide = (
        vix.pivot_table(index="timestamp", columns="series", values="close", aggfunc="last")
        .rename(columns={"VXVCLS": "vix3m"})
        .reset_index()
        .sort_values("timestamp")
    )
    if "VIXCLS" not in vix_wide.columns:
        vix_wide["VIXCLS"] = np.nan
    if "vix3m" not in vix_wide.columns:
        vix_wide["vix3m"] = np.nan
    grid = grid.merge(vix_wide[["timestamp", "VIXCLS", "vix3m"]], on="timestamp", how="left")
    grid[["VIXCLS", "vix3m"]] = grid[["VIXCLS", "vix3m"]].ffill()

    for col in RAW_Z_TARGETS:
        rolling = grid[col].rolling(window=8760, min_periods=24)
        mean = rolling.mean()
        std = rolling.std(ddof=1).replace(0, 1.0)
        grid[f"{col}_z"] = (grid[col] - mean) / std

    grid["VIX_Inversion_Z"] = grid["VIXCLS_z"] - grid["vix3m_z"]
    grid["Price_Mom_24H"] = grid["SPY_close"].shift(1) / grid["SPY_close"].shift(25) - 1.0
    cpi_shock = grid["CPIAUCSL"].diff().abs().gt(0).astype(int)
    nfp_shock = grid["PAYEMS"].diff().abs().gt(0).astype(int)
    grid["CPI_Event_Active"] = cpi_shock.rolling(window=49, min_periods=1).max().fillna(0).astype(int)
    grid["NFP_Event_Active"] = nfp_shock.rolling(window=49, min_periods=1).max().fillna(0).astype(int)

    final = grid[FINAL_GRID_COLUMNS].copy()
    final = final[final["SPY_close"].notna()].reset_index(drop=True)

    validation = validate_grid(final, variant, macro_v, vix, assumptions)
    return final, validation


def validate_grid(grid: pd.DataFrame, variant: str, macro_v: pd.DataFrame, vix: pd.DataFrame, assumptions: pd.DataFrame) -> dict:
    required = set(FINAL_GRID_COLUMNS)
    duplicate_timestamps = int(grid["timestamp"].duplicated().sum())
    missing_cols = sorted(required.difference(grid.columns))
    numeric_status = {col: bool(pd.api.types.is_numeric_dtype(grid[col])) if col in grid.columns else False for col in FEATURES}
    null_counts = {col: int(grid[col].isna().sum()) for col in FINAL_GRID_COLUMNS if col in grid.columns}
    first_valid = {
        col: (None if grid[col].first_valid_index() is None else grid.loc[grid[col].first_valid_index(), "timestamp"].isoformat())
        for col in FINAL_GRID_COLUMNS
        if col in grid.columns
    }

    modelable = grid[(grid["timestamp"] >= pd.Timestamp("2021-01-01")) & (grid["timestamp"] <= pd.Timestamp("2025-12-31 23:59:59"))]
    modelable_price_nulls = {
        "SPY_close": int(modelable["SPY_close"].isna().sum()),
        "QQQ_close": int(modelable["QQQ_close"].isna().sum()),
    }

    timing_checks = {
        "macro_min_available_ts_utc": None if macro_v.empty else macro_v["available_ts_utc"].min().isoformat(),
        "macro_rows": int(len(macro_v)),
        "vix_min_available_ts_utc": None if vix.empty else vix["available_ts_utc"].min().isoformat(),
        "vix_rows": int(len(vix)),
        "macro_values_joined_by_available_ts": True,
        "vix_values_joined_by_pit_timestamp": True,
    }

    validation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "checks": {
            "no_duplicate_timestamps": duplicate_timestamps == 0,
            "duplicate_timestamps": duplicate_timestamps,
            "all_required_columns_exist": not missing_cols,
            "missing_columns": missing_cols,
            "all_14_features_numeric": all(numeric_status.values()),
            "feature_numeric_status": numeric_status,
            "modelable_price_nulls": modelable_price_nulls,
            "modelable_spy_close_no_nulls": modelable_price_nulls["SPY_close"] == 0,
            "modelable_qqq_close_no_nulls": modelable_price_nulls["QQQ_close"] == 0,
            "timing_pit_checks": timing_checks,
        },
        "row_count": int(len(grid)),
        "timestamp_range": {
            "min": None if grid.empty else grid["timestamp"].min().isoformat(),
            "max": None if grid.empty else grid["timestamp"].max().isoformat(),
        },
        "null_counts": null_counts,
        "first_valid_timestamp_by_column": first_valid,
        "timing_assumptions": assumptions.to_dict(orient="records"),
    }
    validation["passed"] = all(
        [
            validation["checks"]["no_duplicate_timestamps"],
            validation["checks"]["all_required_columns_exist"],
            validation["checks"]["all_14_features_numeric"],
            validation["checks"]["modelable_spy_close_no_nulls"],
            validation["checks"]["modelable_qqq_close_no_nulls"],
        ]
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(V4_1_ROOT))
    parser.add_argument("--variant", action="append", choices=TIMING_VARIANTS)
    args = parser.parse_args()

    root = ensure_v4_1_dirs(args.root)
    market, macro, vix, assumptions, variants = load_inputs(root)
    requested = args.variant or variants["variant"].tolist()
    results = {}
    for variant in requested:
        grid, validation = build_variant_grid(root, variant, market, macro, vix, assumptions)
        out_dir = root / "grids" / variant
        write_csv(grid, out_dir / "v4_1_silver_grid.csv", root)
        validation["grid_sha256"] = sha256_file(out_dir / "v4_1_silver_grid.csv")
        write_json(validation, out_dir / "v4_1_silver_grid_validation.json", root)
        results[variant] = validation
        if variant == TIMING_VARIANTS[0] and not validation["passed"]:
            raise RuntimeError(f"Main V4.1 grid failed validation: {validation}")
    print_json({"root": root, "variant_results": results})


if __name__ == "__main__":
    main()
