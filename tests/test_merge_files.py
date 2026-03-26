"""
Unit tests for the merge_files library entry point.
"""

from src.merge_files import run_merge


def test_run_merge_appends_value_column_using_key_mapping(temp_dir):
    """The importable merge helper should append values from file1 onto file2."""
    file1 = temp_dir / "file1.csv"
    file2 = temp_dir / "file2.csv"
    out_file = temp_dir / "merged.csv"

    file1.write_text(
        "Deal Num,Base MTM\n"
        "1001,10.50\n"
        "1002,20.00\n",
        encoding="utf-8-sig",
    )
    file2.write_text(
        "deal_tracking_num,name\n"
        "1001,Alpha\n"
        "1003,Gamma\n",
        encoding="utf-8-sig",
    )

    result = run_merge(
        str(file1),
        str(file2),
        str(out_file),
        [("deal_tracking_num", "Deal Num")],
        ["Base MTM"],
        ["mtm"],
    )

    assert result["matched_keys"] == 2
    assert out_file.read_text(encoding="utf-8-sig").splitlines() == [
        "deal_tracking_num,name,mtm",
        "1001,Alpha,10.50",
        '1003,Gamma,""',
    ]
