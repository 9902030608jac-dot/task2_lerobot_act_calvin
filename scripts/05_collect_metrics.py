#!/usr/bin/env python
"""Collect training/evaluation metrics into report-ready tables and notes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import collect_and_write_report_assets


def main() -> None:
    """Write summary_metrics.csv, summary_metrics.md, and analysis notes."""
    result = collect_and_write_report_assets()
    print(
        json.dumps(
            {
                "summary_metrics_csv": str(result["csv_path"]),
                "summary_metrics_md": str(result["md_path"]),
                "analysis_notes": str(result["notes_path"]),
                "warnings": result["warnings"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
