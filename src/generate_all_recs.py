#!/usr/bin/env python3
"""
Run all trade-attribute reconciliations for a given run folder.

Examples:
  python src/generate_all_recs.py
  python src/generate_all_recs.py --run-folder dr0_run2 --jobs future,pmSwap
  python src/generate_all_recs.py --skip-mtm-merge --skip-qt-end-merge
  python src/generate_all_recs.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RecJob:
    name: str
    config_rel: str
    source_rel_template: str
    output_stem: str
    target_kind: str  # "dm01" or "dm03"


@dataclass(frozen=True)
class MtmMergeJob:
    source_name: str
    output_name: str


@dataclass(frozen=True)
class RunPaths:
    source_trade_dir: Path
    source_mtm_dir: Path
    output_run_mtm_dir: Path
    output_trade_dir: Path
    output_mtm_dir: Path
    output_dm03_dir: Path
    output_recs_dir: Path
    target_dir: Path
    dm01_file: Path
    dm03_raw_file: Path
    dm04_file: Path
    dm03_merged_file: Path
    dm03_merged_txt_file: Path
    mtm_source_file: Path


JOBS: List[RecJob] = [
    RecJob(
        name="future",
        config_rel="config/trade-attributes/future-rec.yaml",
        source_rel_template="output/{run}/trade-attributes/mtm/CME_Futures_GVA_with_mtm.csv",
        output_stem="cmeFuture_rec",
        target_kind="dm03",
    ),
    RecJob(
        name="listedOption",
        config_rel="config/trade-attributes/listedOption-rec.yaml",
        source_rel_template="output/{run}/trade-attributes/mtm/CME_Options_GVA_with_mtm.csv",
        output_stem="cmeOption_rec",
        target_kind="dm03",
    ),
    RecJob(
        name="fxSpotFwd",
        config_rel="config/trade-attributes/fxSpotFwd-rec.yaml",
        source_rel_template="output/{run}/trade-attributes/mtm/FXSpot_Forward_GVA_with_mtm.csv",
        output_stem="fxSpotFwd_rec",
        target_kind="dm01",
    ),
    RecJob(
        name="otcBarrierOption",
        config_rel="config/trade-attributes/otcBarrierOption-rec.yaml",
        source_rel_template="output/{run}/trade-attributes/mtm/Barrier_Options_GVA_with_mtm.csv",
        output_stem="otcBarrierOption_rec",
        target_kind="dm01",
    ),
    RecJob(
        name="otcOption",
        config_rel="config/trade-attributes/otcOption-rec.yaml",
        source_rel_template="output/{run}/trade-attributes/mtm/OTC_Options_GVA_with_mtm.csv",
        output_stem="otcOption_rec",
        target_kind="dm01",
    ),
    RecJob(
        name="pmSpotFwd",
        config_rel="config/trade-attributes/pmSpotFwd-rec.yaml",
        source_rel_template="output/{run}/trade-attributes/mtm/PMSpot_Forward_GVA_with_mtm.csv",
        output_stem="pmSpotFwd_rec",
        target_kind="dm03",
    ),
    RecJob(
        name="pmSwap",
        config_rel="config/trade-attributes/pmSwap-rec.yaml",
        source_rel_template="output/{run}/trade-attributes/mtm/PMSwap_GroupNumber_GVA_with_mtm.csv",
        output_stem="pmSwap_rec",
        target_kind="dm03",
    ),
]

MTM_MERGE_JOBS: List[MtmMergeJob] = [
    MtmMergeJob("Barrier_Options_GVA.csv", "Barrier_Options_GVA_with_mtm.csv"),
    MtmMergeJob("CME Futures_GVA.csv", "CME_Futures_GVA_with_mtm.csv"),
    MtmMergeJob("CME Options_GVA.csv", "CME_Options_GVA_with_mtm.csv"),
    MtmMergeJob("FXSpot_Forward_GVA.csv", "FXSpot_Forward_GVA_with_mtm.csv"),
    MtmMergeJob("OTC Options_GVA.csv", "OTC_Options_GVA_with_mtm.csv"),
    MtmMergeJob("PMSpot_Forward_GVA.csv", "PMSpot_Forward_GVA_with_mtm.csv"),
    MtmMergeJob("PMSwap_GroupNumber_GVA.csv", "PMSwap_GroupNumber_GVA_with_mtm.csv"),
]

MTM_SUMMARY_FILE_NAME = "mtm_by_deal_num_sum.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate trade-attribute reconciliation outputs, including "
            "pre-merge steps for MTM and QT_END."
        )
    )
    parser.add_argument(
        "--run-folder",
        default="dr0_run2",
        help="Folder suffix under source/target/output (default: dr0_run2).",
    )
    parser.add_argument(
        "--jobs",
        default="",
        help=(
            "Comma-separated job names to run. "
            "Default is all jobs. "
            f"Available: {', '.join(job.name for job in JOBS)}"
        ),
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable to invoke t-rex.py with.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level passed to t-rex.py.",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Timestamp override for output names (format YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after first failed reconciliation.",
    )
    parser.add_argument(
        "--skip-mtm-merge",
        action="store_true",
        help="Skip merging MTM into GVA files.",
    )
    parser.add_argument(
        "--skip-qt-end-merge",
        action="store_true",
        help="Skip merging QT_END from dm04 into dm03 target extract.",
    )
    parser.add_argument(
        "--no-dm03-txt-copy",
        action="store_true",
        help="Do not copy merged dm03 CSV to dm03_tp_cm_rep.txt.",
    )
    return parser.parse_args()


def latest_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' under {directory}")
    return matches[0]


def resolve_run_paths(repo_root: Path, run_folder: str) -> RunPaths:
    source_trade_dir = repo_root / "source" / run_folder / "trade-attributes"
    source_mtm_dir = repo_root / "source" / run_folder / "mtm"
    output_run_dir = repo_root / "output" / run_folder
    output_trade_dir = output_run_dir / "trade-attributes"
    output_run_mtm_dir = output_run_dir / "mtm"
    output_mtm_dir = output_trade_dir / "mtm"
    output_dm03_dir = output_trade_dir / "dm03"
    output_recs_dir = output_trade_dir / "recs"
    target_base_dir = repo_root / "target" / run_folder
    target_trade_dir = target_base_dir / "trade-attributes"
    target_dir = target_trade_dir if target_trade_dir.exists() else target_base_dir

    dm01_file = latest_file(target_dir, "dm01_tp_all_rep*.csv")
    dm03_raw_file = latest_file(target_dir, "dm03_tp_cm_rep*.csv")
    dm04_file = latest_file(target_dir, "dm04-dbf_cm_fmat1_rep*.csv")
    dm03_merged_file = output_dm03_dir / "dm03_tp_cm_rep.csv"
    dm03_merged_txt_file = output_dm03_dir / "dm03_tp_cm_rep.txt"
    mtm_source_file = output_run_mtm_dir / MTM_SUMMARY_FILE_NAME

    return RunPaths(
        source_trade_dir=source_trade_dir,
        source_mtm_dir=source_mtm_dir,
        output_run_mtm_dir=output_run_mtm_dir,
        output_trade_dir=output_trade_dir,
        output_mtm_dir=output_mtm_dir,
        output_dm03_dir=output_dm03_dir,
        output_recs_dir=output_recs_dir,
        target_dir=target_dir,
        dm01_file=dm01_file,
        dm03_raw_file=dm03_raw_file,
        dm04_file=dm04_file,
        dm03_merged_file=dm03_merged_file,
        dm03_merged_txt_file=dm03_merged_txt_file,
        mtm_source_file=mtm_source_file,
    )


def select_jobs(jobs_arg: str) -> List[RecJob]:
    if not jobs_arg.strip():
        return JOBS

    job_map = {job.name: job for job in JOBS}
    names = [name.strip() for name in jobs_arg.split(",") if name.strip()]
    unknown = [name for name in names if name not in job_map]
    if unknown:
        raise ValueError(
            f"Unknown job(s): {', '.join(unknown)}. "
            f"Available: {', '.join(job_map.keys())}"
        )
    return [job_map[name] for name in names]


def format_command(parts: List[str]) -> str:
    return subprocess.list2cmdline(parts)


def run_command(cmd: List[str], cwd: Path, dry_run: bool) -> int:
    print(f"    Command: {format_command(cmd)}")
    if dry_run:
        print("    Status: DRY RUN")
        return 0

    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode == 0:
        print("    Status: OK")
    else:
        print(f"    Status: FAILED (exit code {result.returncode})")
    return result.returncode


def normalize_column_name(name: object) -> str:
    return " ".join(str(name).replace("\n", " ").split())


def find_latest_mtm_excel(paths: RunPaths) -> Optional[Path]:
    patterns = ("MTM*.xlsx", "MTM*.xls")
    candidates: List[Path] = []
    for directory in (paths.source_mtm_dir, paths.source_trade_dir):
        if not directory.exists():
            continue
        for pattern in patterns:
            candidates.extend(directory.glob(pattern))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def build_mtm_source_from_excel(
    repo_root: Path, args: argparse.Namespace, paths: RunPaths
) -> bool:
    excel_path = find_latest_mtm_excel(paths)
    if not excel_path:
        print(f"  Missing MTM source file: {paths.mtm_source_file}")
        print(
            "  No MTM Excel file found to build MTM source under "
            f"{paths.source_mtm_dir} or {paths.source_trade_dir}."
        )
        return False

    print("Pre-step 0: Prepare MTM summary source file")
    print(f"  Excel source: {excel_path}")
    print(f"  MTM target: {paths.mtm_source_file}")
    if paths.mtm_source_file.exists():
        print("  Existing MTM summary file will be overwritten.")

    if args.dry_run:
        preview_root = Path(tempfile.gettempdir()) / "mtm_build_preview"
        raw_csv = preview_root / "MTM_deals_raw.csv"
        pivot_merge_out = preview_root / "MTM_deals_raw_merged.csv"
        print("  Status: DRY RUN (would convert Excel to CSV and pivot Base MTM by Deal Num)")

        cmd = [
            args.python_exe,
            "src/merge_files.py",
            "--file1",
            str(raw_csv),
            "--file2",
            str(raw_csv),
            "--out",
            str(pivot_merge_out),
            "--key",
            "Deal Num",
            "--value-col",
            "Base MTM",
            "--out-col",
            "_tmp",
            "--pivot",
            "--pivot-index",
            "Deal Num",
            "--pivot-values",
            "Base MTM",
        ]
        rc = run_command(cmd, repo_root, args.dry_run)
        if rc != 0:
            return False

        print(f"  Would write MTM summary file: {paths.mtm_source_file}")
        print("")
        return True

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        print(f"  Status: FAILED (cannot import pandas: {exc})")
        return False

    try:
        df = pd.read_excel(excel_path, sheet_name=0)
    except Exception as exc:
        print(f"  Status: FAILED (cannot read Excel file: {exc})")
        return False

    df.columns = [normalize_column_name(col) for col in df.columns]
    required = ["Deal Num", "Base MTM"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"  Status: FAILED (missing required MTM columns: {missing})")
        print(f"  Available columns: {list(df.columns)}")
        return False

    with tempfile.TemporaryDirectory(prefix="mtm_build_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        raw_csv = tmp_dir_path / "MTM_deals_raw.csv"
        pivot_merge_out = tmp_dir_path / "MTM_deals_raw_merged.csv"
        pivot_csv = tmp_dir_path / "MTM_deals_raw_pivoted.csv"

        df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        print("  Status: Converted Excel -> CSV")

        cmd = [
            args.python_exe,
            "src/merge_files.py",
            "--file1",
            str(raw_csv),
            "--file2",
            str(raw_csv),
            "--out",
            str(pivot_merge_out),
            "--key",
            "Deal Num",
            "--value-col",
            "Base MTM",
            "--out-col",
            "_tmp",
            "--pivot",
            "--pivot-index",
            "Deal Num",
            "--pivot-values",
            "Base MTM",
        ]
        rc = run_command(cmd, repo_root, args.dry_run)
        if rc != 0:
            return False

        if not pivot_csv.exists():
            print(f"  Status: FAILED (expected pivot output not found: {pivot_csv})")
            return False

        paths.mtm_source_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pivot_csv, paths.mtm_source_file)

    print(f"  Status: Wrote MTM summary file: {paths.mtm_source_file}")
    print("")
    return True


def run_mtm_merge_step(repo_root: Path, args: argparse.Namespace, paths: RunPaths) -> bool:
    print("Pre-step 1: Merge MTM into GVA files")
    if not build_mtm_source_from_excel(repo_root, args, paths):
        return False

    for index, job in enumerate(MTM_MERGE_JOBS, start=1):
        file2_path = paths.source_trade_dir / job.source_name
        out_path = paths.output_mtm_dir / job.output_name

        print(f"  [{index}/{len(MTM_MERGE_JOBS)}] {job.output_name}")
        print(f"    File1 (MTM): {paths.mtm_source_file}")
        print(f"    File2 (GVA): {file2_path}")
        print(f"    Out: {out_path}")

        if not file2_path.exists():
            print(f"    Status: SKIPPED (missing file2: {file2_path})")
            continue

        cmd = [
            args.python_exe,
            "src/merge_files.py",
            "--file1",
            str(paths.mtm_source_file),
            "--file2",
            str(file2_path),
            "--out",
            str(out_path),
            "--key",
            "deal_tracking_num=Deal Num",
            "--value-col",
            "Base MTM",
            "--out-col",
            "mtm",
        ]
        rc = run_command(cmd, repo_root, args.dry_run)
        if rc != 0:
            return False
    print("")
    return True


def run_qt_end_merge_step(repo_root: Path, args: argparse.Namespace, paths: RunPaths) -> bool:
    print("Pre-step 2: Merge QT_END from dm04 into dm03")
    print(f"  File1 (dm04): {paths.dm04_file}")
    print(f"  File2 (dm03 raw): {paths.dm03_raw_file}")
    print(f"  Out (dm03 merged): {paths.dm03_merged_file}")

    cmd = [
        args.python_exe,
        "src/merge_files.py",
        "--file1",
        str(paths.dm04_file),
        "--file2",
        str(paths.dm03_raw_file),
        "--out",
        str(paths.dm03_merged_file),
        "--key",
        "INSTRUMENT",
        "--key",
        "TP_CMCMAT=LABEL",
        "--value-col",
        "QT_END",
        "--out-col",
        "QT_END",
    ]
    rc = run_command(cmd, repo_root, args.dry_run)
    if rc != 0:
        return False

    if args.no_dm03_txt_copy:
        print("  Skipping dm03_tp_cm_rep.txt copy (--no-dm03-txt-copy).")
        print("")
        return True

    txt_path = paths.dm03_merged_txt_file
    if args.dry_run:
        print(f"  Would copy CSV -> TXT: {txt_path}")
    else:
        shutil.copyfile(paths.dm03_merged_file, txt_path)
        print(f"  Copied CSV -> TXT: {txt_path}")
    print("")
    return True


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_folder = args.run_folder
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        selected_jobs = select_jobs(args.jobs)
        paths = resolve_run_paths(repo_root, run_folder)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1

    # Ensure output folders exist
    paths.output_run_mtm_dir.mkdir(parents=True, exist_ok=True)
    paths.output_trade_dir.mkdir(parents=True, exist_ok=True)
    paths.output_mtm_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dm03_dir.mkdir(parents=True, exist_ok=True)
    paths.output_recs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run folder: {run_folder}")
    print(f"MTM source dir: {paths.output_run_mtm_dir}")
    print(f"MTM source file: {paths.mtm_source_file}")
    print(f"MTM output dir: {paths.output_mtm_dir}")
    print(f"dm03 output dir: {paths.output_dm03_dir}")
    print(f"rec output dir: {paths.output_recs_dir}")
    print(f"dm01 target: {paths.dm01_file}")
    print(f"dm03 raw target: {paths.dm03_raw_file}")
    print(f"dm04 source for QT_END: {paths.dm04_file}")
    print(f"target input dir: {paths.target_dir}")
    print(f"dm03 merged target path: {paths.dm03_merged_file}")
    print(f"Jobs: {', '.join(job.name for job in selected_jobs)}")
    print("")

    if not args.skip_mtm_merge:
        ok = run_mtm_merge_step(repo_root, args, paths)
        if not ok:
            return 1
    else:
        print("Pre-step 1 skipped: MTM merge")
        print("")

    if not args.skip_qt_end_merge:
        ok = run_qt_end_merge_step(repo_root, args, paths)
        if not ok:
            return 1
    else:
        print("Pre-step 2 skipped: QT_END merge")
        print("")

    dm03_target_for_recs = (
        paths.dm03_merged_file
        if (paths.dm03_merged_file.exists() or (args.dry_run and not args.skip_qt_end_merge))
        else paths.dm03_raw_file
    )
    targets: Dict[str, Path] = {
        "dm01": paths.dm01_file,
        "dm03": dm03_target_for_recs,
    }
    print(f"Using dm01 for recs: {targets['dm01']}")
    print(f"Using dm03 for recs: {targets['dm03']}")
    print("")

    missing_configs: List[str] = []
    skipped_missing_sources: List[str] = []
    runnable_jobs: List[RecJob] = []
    mtm_outputs_expected = {
        (paths.output_mtm_dir / merge_job.output_name).resolve() for merge_job in MTM_MERGE_JOBS
    }
    for job in selected_jobs:
        config_path = repo_root / job.config_rel
        source_path = repo_root / job.source_rel_template.format(run=run_folder)
        if not config_path.exists():
            missing_configs.append(f"[{job.name}] missing config: {config_path}")
            continue
        if not source_path.exists():
            source_resolved = source_path.resolve()
            source_is_expected_mtm = source_resolved in mtm_outputs_expected
            will_be_produced_by_mtm_step = source_is_expected_mtm and not args.skip_mtm_merge
            if args.dry_run and will_be_produced_by_mtm_step:
                runnable_jobs.append(job)
                continue
            skipped_missing_sources.append(f"[{job.name}] missing source: {source_path}")
            continue
        runnable_jobs.append(job)

    if missing_configs:
        print("Input validation failed:")
        for issue in missing_configs:
            print(f"  - {issue}")
        return 1

    if skipped_missing_sources:
        print("Skipping jobs with missing source inputs:")
        for issue in skipped_missing_sources:
            print(f"  - {issue}")
        print("")

    if not runnable_jobs:
        print("No runnable jobs after input validation.")
        return 0

    failures: List[str] = []
    generated_files: List[Path] = []

    for index, job in enumerate(runnable_jobs, start=1):
        config_path = repo_root / job.config_rel
        source_path = repo_root / job.source_rel_template.format(run=run_folder)
        target_path = targets[job.target_kind]
        output_path = paths.output_recs_dir / f"{job.output_stem}_{timestamp}.xlsx"

        cmd = [
            args.python_exe,
            "t-rex.py",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--log-level",
            args.log_level,
        ]

        print(f"[{index}/{len(runnable_jobs)}] {job.name}")
        print(f"  Source: {source_path}")
        print(f"  Target: {target_path}")
        print(f"  Output: {output_path}")
        rc = run_command(cmd, repo_root, args.dry_run)
        if rc == 0:
            if not args.dry_run:
                generated_files.append(output_path)
        else:
            failures.append(job.name)
            if args.stop_on_error:
                print("")
                break
        print("")

    print("Summary")
    if args.dry_run:
        print(f"  Planned jobs: {len(runnable_jobs) - len(failures)}")
    else:
        print(f"  Successful: {len(generated_files)}")
    print(f"  Failed: {len(failures)}")
    if skipped_missing_sources:
        print(f"  Skipped (missing source): {len(skipped_missing_sources)}")

    if generated_files:
        print("  Generated files:")
        for file_path in generated_files:
            print(f"    - {file_path}")

    if failures:
        print(f"  Failed jobs: {', '.join(failures)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
