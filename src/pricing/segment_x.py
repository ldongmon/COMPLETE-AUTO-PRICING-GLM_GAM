"""
segment_x.py
============
The "Segment X" of the business question, selected in Phase 8 and reused by
Phases 9 to 11 and the app. Phase 8 writes its definition and measured cost
relativity to `reports/figures/segment_x.json`; later phases load that file
rather than retyping any number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SEGMENT_X_FILE = Path("reports") / "figures" / "segment_x.json"


def save_segment_x(spec: dict, project_root: Path) -> Path:
    path = project_root / SEGMENT_X_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def load_segment_x(project_root: Path) -> dict:
    return json.loads((project_root / SEGMENT_X_FILE).read_text(encoding="utf-8"))


def in_segment_x(df: pd.DataFrame, spec: dict) -> pd.Series:
    """Boolean mask: rows whose `spec['column']` lies in [spec['low'], spec['high']]."""
    return df[spec["column"]].between(spec["low"], spec["high"])
