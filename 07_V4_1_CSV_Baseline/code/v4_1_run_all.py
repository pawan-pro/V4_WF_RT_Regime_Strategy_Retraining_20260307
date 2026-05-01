"""Run the isolated V4.1 CSV-only pipeline end to end."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

VOLUME_CODE_DIR = Path("/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/code")
DBFS_CODE_DIR = Path("/dbfs/Volumes/pawan_quant_research_1/v4/market_data/v4_1_csv/code")
THIS_DIR = Path(globals().get("__file__", ".")).resolve().parent
for candidate in [VOLUME_CODE_DIR, DBFS_CODE_DIR, THIS_DIR]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from v4_1_common import V4_1_ROOT, copy_code_to_volume, ensure_v4_1_dirs, print_json


def code_dir() -> Path:
    configured = os.environ.get("V4_1_CODE_DIR")
    if configured and Path(configured).exists():
        return Path(configured)
    if VOLUME_CODE_DIR.exists():
        return VOLUME_CODE_DIR
    if DBFS_CODE_DIR.exists():
        return DBFS_CODE_DIR
    return THIS_DIR


def run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(code_dir() / script), *args]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(V4_1_ROOT))
    parser.add_argument("--fred-api-key", default=None)
    parser.add_argument("--skip-inputs", action="store_true")
    args = parser.parse_args()
    root = ensure_v4_1_dirs(args.root)
    copy_code_to_volume(root)
    common_args = ["--root", str(root)]
    if not args.skip_inputs:
        input_args = [*common_args]
        if args.fred_api_key:
            input_args += ["--fred-api-key", args.fred_api_key]
        run("v4_1_setup_inputs_from_sources.py", input_args)
    run("v4_1_build_grid_from_csv.py", common_args)
    run("v4_1_train_walkforward_from_csv.py", common_args)
    run("v4_1_backtest_performance_from_csv.py", common_args)
    run("v4_1_write_reports.py", common_args)
    print_json({"completed": True, "root": root})


if __name__ == "__main__":
    main()
