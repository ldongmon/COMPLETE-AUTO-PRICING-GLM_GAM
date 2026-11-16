"""
segment_x.py
============
"Segment X" of the business question: a group of policies defined by a range
of one rating factor, here BonusMalus between 51 and 80.

Phase 8 finds it and writes its definition and measured cost relativity to
reports/figures/segment_x.json. Phases 9 to 11 and the app read that file, so
no number is ever retyped.
"""

import json  # reads and writes the definition file
from pathlib import Path  # file paths

SEGMENT_X_FILE = Path("reports") / "figures" / "segment_x.json"  # where the definition is stored


def save_segment_x(spec, project_root):
    """Write the definition (a dict) to reports/figures/segment_x.json."""
    path = project_root / SEGMENT_X_FILE
    path.parent.mkdir(parents=True, exist_ok=True)  # create the folder if needed
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def load_segment_x(project_root):
    """Read the definition written by Phase 8."""
    return json.loads((project_root / SEGMENT_X_FILE).read_text(encoding="utf-8"))


def in_segment_x(df, spec):
    """True for the policies in the segment: low <= value <= high (both limits included)."""
    return df[spec["column"]].between(spec["low"], spec["high"])
