# Databricks notebook source
"""Finalize V4.2 PIT teacher-student research artifacts.

This script does not train models. It reads the already-written research CSV
outputs and writes only summary/manifest files under the isolated research
output root.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CATALOG = "pawan_quant_research_1"
SCHEMA = "v4"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_bt_master_grid_1h"
PRODUCTION_SIGNAL_TABLE = f"{CATALOG}.{SCHEMA}.bt_walkforward_regime_signals"
OUTPUT_ROOT = Path(f"/Volumes/{CATALOG}/{SCHEMA}/market_data/v4_2_pit_teacher_student_2026-05-04")
PERFORMANCE_DIR = OUTPUT_ROOT / "performance"
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
FORENSICS_DIR = OUTPUT_ROOT / "forensics"
CODE_PATH = OUTPUT_ROOT / "code" / "01_V4_2_PIT_Teacher_Student.py"


def ensure_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openpyxl"])


def assert_research_output_path(path: Path) -> None:
    if not str(path).startswith(str(OUTPUT_ROOT)):
        raise ValueError(f"Refusing write outside research output root: {path}")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(payload: dict[str, Any], path: Path) -> None:
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return None if math.isnan(float(obj)) else float(obj)
        return obj

    assert_research_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    out = df.copy().astype(object).where(pd.notna(df), "")
    cols = [str(c) for c in out.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in out.itertuples(index=False):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ensure_openpyxl()
    summary = pd.read_csv(PERFORMANCE_DIR / "variant_summary.csv")
    diagnostics = pd.read_csv(PERFORMANCE_DIR / "scaling_readiness_diagnostics.csv")

    audit_path = PERFORMANCE_DIR / "audit_pack.xlsx"
    assert_research_output_path(audit_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_audit_path = Path(tmpdir) / "audit_pack.xlsx"
        with pd.ExcelWriter(tmp_audit_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            diagnostics.to_excel(writer, sheet_name="scaling_diagnostics", index=False)
            for variant_dir in sorted([p for p in PERFORMANCE_DIR.iterdir() if p.is_dir()]):
                annual = variant_dir / "annual_returns.csv"
                trailing = variant_dir / "trailing_metrics.csv"
                if annual.exists():
                    pd.read_csv(annual).to_excel(writer, sheet_name=f"annual_{variant_dir.name[:20]}", index=False)
                if trailing.exists():
                    pd.read_csv(trailing).to_excel(writer, sheet_name=f"trailing_{variant_dir.name[:18]}", index=False)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_audit_path, audit_path)

    runtime_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "spark_runtime": os.environ.get("DATABRICKS_RUNTIME_VERSION"),
        "source_table": SILVER_TABLE,
        "read_only_reference_table": PRODUCTION_SIGNAL_TABLE,
        "output_root": OUTPUT_ROOT,
        "research_script_path": CODE_PATH,
        "research_script_sha256": file_sha256(CODE_PATH),
        "variant_summary_sha256": file_sha256(PERFORMANCE_DIR / "variant_summary.csv"),
        "fold_manifest_sha256": file_sha256(MANIFESTS_DIR / "fold_manifest.csv"),
        "audit_pack_sha256": file_sha256(audit_path),
        "production_delta_writes": False,
        "model_registry_writes": False,
        "saxo_or_rt_writes": False,
        "workflow_or_schedule_changes": False,
        "finalizer_trained_models": False,
    }
    write_json(runtime_manifest, MANIFESTS_DIR / "runtime_manifest.json")

    lines = [
        "# V4.2 PIT Teacher-Student Research Summary",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Output root: `{OUTPUT_ROOT}`",
        f"- Source table: `{SILVER_TABLE}` read only",
        "- Production writes: none",
        "- Model registry writes: none",
        "- Saxo/RT/dashboard/workflow changes: none",
        "",
        "## Variant Summary",
        "",
        md_table(summary),
        "",
        "## Governance",
        "",
        "The retrospective full-sample teacher reference is marked `research_only` by construction because it is not strict PIT. No promotion was performed.",
        "",
    ]
    summary_path = FORENSICS_DIR / "pit_teacher_student_research_summary.md"
    assert_research_output_path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Finalized: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
