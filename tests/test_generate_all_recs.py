"""
Focused tests for generate_all_recs helpers.
"""

import pandas as pd

from src.generate_all_recs import build_mtm_summary_file


def test_build_mtm_summary_file_aggregates_base_mtm(temp_dir):
    """MTM summary generation should normalize headers and aggregate by Deal Num."""
    excel_path = temp_dir / "mtm.xlsx"
    output_path = temp_dir / "mtm_summary.csv"

    pd.DataFrame(
        {
            "Deal\nNum": ["1001", "1001", "1002", " 1002 ", ""],
            "Base MTM": [10.5, 1.5, 3.0, 2.0, 99.0],
        }
    ).to_excel(excel_path, index=False)

    build_mtm_summary_file(excel_path, output_path)

    result = pd.read_csv(output_path, dtype={"Deal Num": str})
    assert list(result.columns) == ["Deal Num", "Base MTM"]
    assert result.to_dict("records") == [
        {"Deal Num": "1001", "Base MTM": 12.0},
        {"Deal Num": "1002", "Base MTM": 5.0},
    ]
