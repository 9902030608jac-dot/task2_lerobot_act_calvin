#!/usr/bin/env python
"""Plot training and D-evaluation figures for the report."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import plot_all_report_figures


def main() -> None:
    """Create all available report figures."""
    outputs = plot_all_report_figures()
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
