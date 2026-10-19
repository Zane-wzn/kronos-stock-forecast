"""Strict Page1-C Static versus MARS-CL runner. Run --dry-run before seed 100."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.kronos_continual.runner import main

if __name__ == "__main__":
    main()
