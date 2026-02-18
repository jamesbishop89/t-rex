#!/usr/bin/env python3
"""
Generic CSV merge: append one or more value columns from File 1 onto File 2,
matching rows on one or more key columns.

Key goals:
- Preserve File 2 row formatting exactly (including decimal formatting / trailing zeros),
  by copying each input line verbatim and appending the extra field(s).
- Flexible: works for any column (MTM, price, quantity, etc.).
- Optional pivot/aggregation of File 1 before merging

Assumptions:
- File 2 is "one CSV record per physical line" (no embedded newlines inside quoted fields).

Usage:
  python merge_files.py --file1 file1.csv --file2 file2.csv --out .\\output\\merged.csv

Optional:
  --key tran_num            Key column(s) (default: tran_num). Repeatable / comma-separated.
                            When the column has the SAME name in both files:
                              --key tran_num --key currency
                            When the column has DIFFERENT names (file2=file1):
                              --key INSTRUMENT --key TP_CMCMAT=LABEL
  --value-col MTM           Column(s) in File 1 to bring across (default: MTM).
                            May be repeated or comma-separated.
  --out-col mtm             Output column name(s) in merged file. If omitted the
                            value-col name(s) are lowered. May be repeated/comma-separated
                            and must match --value-col count when given.
  --encoding utf-8-sig      File encoding (default: utf-8-sig)
  
Pivot/Aggregation:
  --pivot                   Enable pivot/aggregation of file1 before merging
  --pivot-index deal_num    Column(s) to group by (repeatable/comma-separated)
  --pivot-values BASE_MTM   Column(s) to aggregate (repeatable/comma-separated)
  --pivot-aggfunc sum       Aggregation function: sum, mean, first, last, min, max (default: sum)
"""

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _parse_list(raw: List[str]) -> List[str]:
    """Flatten repeated / comma-separated CLI values into a flat list."""
    out: List[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _parse_key_pairs(raw: List[str]) -> List[Tuple[str, str]]:
    """Parse key specs into (file2_col, file1_col) pairs.

    Supports:
      --key col             -> (col, col)           same name in both files
      --key f2col=f1col     -> (f2col, f1col)       different names
      --key a,b=c           -> [(a, a), (b, c)]     comma-separated mix
    """
    pairs: List[Tuple[str, str]] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                f2, f1 = part.split("=", 1)
                pairs.append((f2.strip(), f1.strip()))
            else:
                pairs.append((part, part))
    return pairs


def _make_composite_key(row_or_fields, key_cols, *, by_index: bool = False) -> str:
    """Build a single lookup string from one or more column values.

    If *by_index* is True, *key_cols* should be a list of int indices and
    *row_or_fields* a list of field strings.
    Otherwise *key_cols* is a list of resolved column names and *row_or_fields*
    is a csv.DictReader row.
    """
    parts: List[str] = []
    if by_index:
        for idx in key_cols:
            val = (row_or_fields[idx] if idx < len(row_or_fields) else "").strip()
            parts.append(val)
    else:
        for col in key_cols:
            val = (row_or_fields.get(col, "") or "").strip()
            parts.append(val)
    return "\x00".join(parts)  # null-byte separator avoids collisions


def sniff_dialect(path: str, encoding: str) -> csv.Dialect:
    """Best-effort sniff; falls back to csv.excel (comma)."""
    with open(path, "r", newline="", encoding=encoding) as f:
        sample = f.read(65536)
    try:
        return csv.Sniffer().sniff(sample)
    except csv.Error:
        return csv.excel


def resolve_column(fieldnames: List[str], requested: str) -> str:
    """Resolve requested column name by exact or case-insensitive match."""
    if requested in fieldnames:
        return requested
    req_n = _norm(requested)
    for f in fieldnames:
        if _norm(f) == req_n:
            return f
    raise ValueError(
        f"Missing required column '{requested}'. Found columns: {fieldnames}"
    )


def serialize_one_field(value: str, dialect: csv.Dialect) -> str:
    """
    Serialize a single CSV field using the same dialect (quotes if needed).
    Returns just the field text (no trailing newline).
    """
    buf = io.StringIO()
    w = csv.writer(buf, dialect=dialect)
    w.writerow([value])
    return buf.getvalue().rstrip("\r\n")


def pivot_file1(
    file_path: str,
    pivot_index: List[str],
    pivot_values: List[str],
    aggfunc: str,
    encoding: str,
) -> str:
    """
    Pivot/aggregate file1 using pandas and return path to temporary pivoted file.
    
    Args:
        file_path: Path to the CSV file to pivot
        pivot_index: Column(s) to group by
        pivot_values: Column(s) to aggregate
        aggfunc: Aggregation function (sum, mean, first, last, min, max)
        encoding: File encoding
        
    Returns:
        Path to temporary pivoted CSV file
    """
    # Read the file
    dialect = sniff_dialect(file_path, encoding)
    df = pd.read_csv(file_path, encoding=encoding, dialect=dialect)
    
    print(f"Original file1 shape: {df.shape}", file=sys.stderr)
    
    # Ensure requested columns exist
    missing_idx = [col for col in pivot_index if col not in df.columns]
    missing_val = [col for col in pivot_values if col not in df.columns]
    if missing_idx:
        raise ValueError(f"Pivot index column(s) not found in file1: {missing_idx}")
    if missing_val:
        raise ValueError(f"Pivot value column(s) not found in file1: {missing_val}")
    
    # Build aggregation dict
    agg_dict = {col: aggfunc for col in pivot_values}
    
    # Group and aggregate
    grouped = df.groupby(pivot_index, as_index=False).agg(agg_dict)
    
    print(f"After aggregation shape: {grouped.shape}", file=sys.stderr)
    print(f"Aggregated {len(df)} rows to {len(grouped)} rows using {aggfunc}", file=sys.stderr)
    
    # Write to temporary file
    temp_path = file_path.rsplit(".", 1)[0] + "_pivoted.csv"
    grouped.to_csv(temp_path, index=False, encoding=encoding)
    
    return temp_path


def load_value_map(
    file1_path: str,
    file1_key_cols: List[str],
    value_cols: List[str],
    encoding: str,
) -> Dict[str, List[str]]:
    """Read File 1 and build {composite_key: [val1, val2, ...]} lookup."""
    dialect1 = sniff_dialect(file1_path, encoding)

    val_map: Dict[str, List[str]] = {}
    dupes = 0

    with open(file1_path, "r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, dialect=dialect1)
        if not reader.fieldnames:
            raise ValueError(f"{file1_path} appears to have no header row.")

        resolved_keys = [resolve_column(reader.fieldnames, k) for k in file1_key_cols]
        resolved_vals = [resolve_column(reader.fieldnames, v) for v in value_cols]

        for _line_no, row in enumerate(reader, start=2):
            key = _make_composite_key(row, resolved_keys)
            if not key.replace("\x00", ""):
                continue

            # Keep values EXACTLY as parsed (no float conversions)
            vals = [row.get(vc, "") for vc in resolved_vals]
            if key in val_map:
                dupes += 1
            val_map[key] = vals

    if dupes:
        print(f"Warning: {dupes} duplicate key(s) in file1; last value wins.", file=sys.stderr)

    return val_map


def merge_files(
    file2_path: str,
    out_path: str,
    val_map: Dict[str, List[str]],
    file2_key_cols: List[str],
    out_col_names: List[str],
    encoding: str,
) -> None:
    dialect2 = sniff_dialect(file2_path, encoding)
    delim = dialect2.delimiter

    out_parent = Path(out_path).parent
    if str(out_parent) not in ("", "."):
        out_parent.mkdir(parents=True, exist_ok=True)

    with open(file2_path, "r", encoding=encoding, newline="") as fin, open(
        out_path, "w", encoding=encoding, newline=""
    ) as fout:

        header_line = fin.readline()
        if not header_line:
            raise ValueError(f"{file2_path} is empty.")

        # Preserve newline style from header
        newline_seq = "\r\n" if header_line.endswith("\r\n") else "\n"

        header_no_nl = header_line.rstrip("\r\n")
        header_fields = next(csv.reader([header_no_nl], dialect=dialect2))

        # Find key column indices (support composite keys)
        key_indices: List[int] = []
        for kc in file2_key_cols:
            found = False
            for i, name in enumerate(header_fields):
                if _norm(name) == _norm(kc):
                    key_indices.append(i)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"{file2_path} missing required key column '{kc}'. Found columns: {header_fields}"
                )

        # Ensure we are not double-adding any output column
        existing_normed = {_norm(c) for c in header_fields}
        for oc in out_col_names:
            if _norm(oc) in existing_normed:
                raise ValueError(
                    f"{file2_path} already contains column '{oc}' (case-insensitive). "
                    "Remove it or rename it before merging."
                )

        # Write header verbatim + new column name(s)
        fout.write(header_no_nl + delim + delim.join(out_col_names) + newline_seq)

        # Process rows line-by-line, preserve formatting of File 2 exactly
        for raw_line in fin:
            if raw_line == "":
                continue

            line_no_nl = raw_line.rstrip("\r\n")

            # Skip truly blank lines (optional)
            if line_no_nl == "":
                continue

            fields = next(csv.reader([line_no_nl], dialect=dialect2))
            if any(idx >= len(fields) for idx in key_indices):
                # Row is malformed — append blank values
                vals = [""] * len(out_col_names)
            else:
                key = _make_composite_key(fields, key_indices, by_index=True)
                vals = val_map.get(key, [""] * len(out_col_names)) if key.replace("\x00", "") else [""] * len(out_col_names)

            serialized = delim.join(serialize_one_field(v, dialect2) for v in vals)
            fout.write(line_no_nl + delim + serialized + newline_seq)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generic CSV merge: append value column(s) from file1 onto file2, matching on key column(s)."
    )
    ap.add_argument("--file1", required=True, help="Source CSV containing key + value columns")
    ap.add_argument("--file2", required=True, help="Target CSV to receive the extra column(s)")
    ap.add_argument("--out", required=True, help="Output CSV path")

    ap.add_argument("--key", action="append", default=None,
                    help="Key column(s) for joining (default: tran_num). "
                         "Repeatable / comma-separated. Use file2_col=file1_col "
                         "when the key has different names in each file, e.g. "
                         "--key INSTRUMENT --key TP_CMCMAT=LABEL")
    ap.add_argument("--value-col", action="append", default=None,
                    help="Column(s) in file1 to bring across (default: MTM). "
                         "Repeatable / comma-separated.")
    ap.add_argument("--out-col", action="append", default=None,
                    help="Output column name(s) in merged file. If omitted, "
                         "value-col name(s) are lowered. Repeatable / comma-separated; "
                         "must match --value-col count when given.")
    ap.add_argument("--encoding", default="utf-8-sig", help="File encoding (default: utf-8-sig)")
    
    # Pivot/aggregation arguments
    ap.add_argument("--pivot", action="store_true",
                    help="Enable pivot/aggregation of file1 before merging")
    ap.add_argument("--pivot-index", action="append", default=None,
                    help="Column(s) to group by for aggregation. Repeatable / comma-separated.")
    ap.add_argument("--pivot-values", action="append", default=None,
                    help="Column(s) to aggregate. Repeatable / comma-separated.")
    ap.add_argument("--pivot-aggfunc", default="sum",
                    choices=["sum", "mean", "first", "last", "min", "max"],
                    help="Aggregation function (default: sum)")
    
    args = ap.parse_args()

    key_pairs = _parse_key_pairs(args.key) if args.key else [("tran_num", "tran_num")]
    file2_keys = [p[0] for p in key_pairs]
    file1_keys = [p[1] for p in key_pairs]

    value_cols = _parse_list(args.value_col) if args.value_col else ["MTM"]
    out_cols = _parse_list(args.out_col) if args.out_col else [v.lower() for v in value_cols]

    if len(out_cols) != len(value_cols):
        ap.error(f"--out-col count ({len(out_cols)}) must match --value-col count ({len(value_cols)})")

    # Handle pivot/aggregation if requested
    file1_to_use = args.file1
    if args.pivot:
        if not args.pivot_index:
            ap.error("--pivot requires --pivot-index to be specified")
        if not args.pivot_values:
            ap.error("--pivot requires --pivot-values to be specified")
        
        pivot_index = _parse_list(args.pivot_index)
        pivot_values = _parse_list(args.pivot_values)
        
        print(f"Pivoting file1 by {pivot_index}, aggregating {pivot_values} using {args.pivot_aggfunc}...", file=sys.stderr)
        file1_to_use = pivot_file1(args.file1, pivot_index, pivot_values, args.pivot_aggfunc, args.encoding)

    val_map = load_value_map(file1_to_use, file1_keys, value_cols, args.encoding)
    merge_files(args.file2, args.out, val_map, file2_keys, out_cols, args.encoding)

    print(f"Matched keys from file1: {len(val_map)}", file=sys.stderr)
    key_desc = ', '.join(f'{f2}={f1}' if f2 != f1 else f2 for f2, f1 in key_pairs)
    print(f"Key column(s): {key_desc}", file=sys.stderr)
    print(f"Value column(s): {', '.join(value_cols)} -> {', '.join(out_cols)}", file=sys.stderr)
    print(f"Wrote merged file: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())