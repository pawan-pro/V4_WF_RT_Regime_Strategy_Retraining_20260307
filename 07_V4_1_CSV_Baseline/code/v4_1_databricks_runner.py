# Databricks notebook source
"""Notebook-task runner for the isolated V4.1 CSV-only pipeline."""

import runpy
import subprocess
import sys
import importlib.util
from pathlib import Path


try:
    _widgets = dbutils.widgets.getAll()
except Exception:
    _widgets = {}

ROOT = _widgets.get("root") or "/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv"
FRED_API_KEY = _widgets.get("fred_api_key") or ""
MODE = _widgets.get("mode") or "all"
VARIANT = _widgets.get("variant") or ""
CODE_DIR = f"{ROOT}/code"

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

if MODE in {"all", "backtest", "report"} and importlib.util.find_spec("openpyxl") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])


def run_script(script_name, args):
    old_argv = sys.argv[:]
    try:
        sys.argv = [f"{CODE_DIR}/{script_name}", *args]
        runpy.run_path(
            f"{CODE_DIR}/{script_name}",
            run_name="__main__",
            init_globals={"spark": spark, "dbutils": dbutils},
        )
    finally:
        sys.argv = old_argv


common_args = ["--root", ROOT]
variant_args = [*common_args]
if VARIANT:
    variant_args += ["--variant", VARIANT]
setup_args = [*common_args]
if FRED_API_KEY:
    setup_args += ["--fred-api-key", FRED_API_KEY]

input_manifest = Path(ROOT) / "manifests" / "v4_1_input_hashes.json"
if MODE in {"all", "inputs"}:
    if input_manifest.exists():
        print(f"Skipping input freeze; existing manifest found: {input_manifest}")
    else:
        run_script("v4_1_setup_inputs_from_sources.py", setup_args)
if MODE in {"all", "build"}:
    run_script("v4_1_build_grid_from_csv.py", variant_args)
if MODE in {"all", "train"}:
    run_script("v4_1_train_walkforward_from_csv.py", variant_args)
if MODE in {"all", "backtest"}:
    run_script("v4_1_backtest_performance_from_csv.py", variant_args)
if MODE in {"all", "report"}:
    run_script("v4_1_write_reports.py", common_args)

print({"completed": True, "root": ROOT, "mode": MODE, "variant": VARIANT})
