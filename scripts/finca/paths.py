"""Resolve the archive layout used by the FinCA experiments."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("FINCA_RUNS", ROOT / "runs" / "finca")).resolve()
OUTPUT = Path(os.environ.get("FINCA_OUTPUT", RUNS / "adjusted_ablation_20261125")).resolve()
DATA = Path(os.environ.get("FINCA_MARKET_DATA", ROOT / "delete" / "files" / "date")).resolve()
ARCHIVED_RESULTS = Path(os.environ.get("FINCA_ARCHIVED_RESULTS", ROOT / "delete" / "files" / "result")).resolve()


def unique_path(directory, pattern):
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one archived path for {pattern!r} in {directory}, found {len(matches)}")
    return matches[0]
