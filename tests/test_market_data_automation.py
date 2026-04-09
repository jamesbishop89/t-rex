"""
Tests for market-data automation helpers and orchestration.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
from types import SimpleNamespace

from src.market_data_automation import (
    AutomationState,
    EmailSettings,
    MarketDataAutomationService,
    MarketDataJobSpec,
    collect_file_candidates,
    load_job_specs,
    parse_group_selection,
    parse_timestamp_from_name,
    select_pending_pair,
)


def test_parse_timestamp_from_name_reads_target_style_timestamp():
    """Target file names should expose their embedded extraction timestamp."""
    parsed = parse_timestamp_from_name("md_fxvol_rep_eod_20260327_200243.csv")
    assert parsed == datetime(2026, 3, 27, 20, 2, 43)


def test_collect_file_candidates_supports_nested_processedfiles_globs(temp_dir):
    """Source discovery should work when files live under a ProcessedFiles subdirectory."""
    source_root = temp_dir / "source"
    nested_dir = source_root / "fxsp" / "ProcessedFiles"
    nested_dir.mkdir(parents=True)

    older_name = nested_dir / "FXSP_MUREX_EOD-DATA_TODAY_1.csv"
    newer_name = nested_dir / "FXSP_MUREX_EOD-DATA_TODAY_2.csv"
    ignored_name = nested_dir / "FXSP_MUREX_EOD-DATA_TODAY_3.tmp"

    older_name.write_text("header\n", encoding="utf-8")
    newer_name.write_text("header\n", encoding="utf-8")
    ignored_name.write_text("header\n", encoding="utf-8")

    now = datetime(2026, 3, 27, 23, 0, 0)
    os.utime(older_name, (now.timestamp() - 10, now.timestamp() - 10))
    os.utime(newer_name, (now.timestamp() - 100, now.timestamp() - 100))
    os.utime(ignored_name, (now.timestamp() - 100, now.timestamp() - 100))

    candidates = collect_file_candidates(
        directory=source_root,
        include_globs=["fxsp/ProcessedFiles/FXSP_MUREX_EOD-DATA_TODAY_*"],
        exclude_globs=["*.tmp"],
        min_file_age_seconds=0,
        now=now,
    )

    assert [candidate.path.name for candidate in candidates] == [
        "FXSP_MUREX_EOD-DATA_TODAY_1.csv",
        "FXSP_MUREX_EOD-DATA_TODAY_2.csv",
    ]


def test_load_job_specs_filters_by_group(temp_dir):
    """Job groups should let us split intraday and eod schedules cleanly."""
    config_file = temp_dir / "automation.yaml"
    config_file.write_text(
        """
jobs:
  - name: fxsp
    groups: [intraday]
    config: fxsp-rec.yaml
    source_globs: [fxsp/ProcessedFiles/FXSP_MUREX_EOD-DATA_TODAY_*.csv]
    target_globs: [fxrate_rep_itd_*.csv]
  - name: fxsp_eod
    groups: [eod]
    config: fxsp-rec.yaml
    source_globs: [fxsp_eod/ProcessedFiles/FXSP_MUREX_EOD-DATA_TODAY_*.csv]
    target_globs: [fxrate_rep_eod_*.csv]
""".strip(),
        encoding="utf-8",
    )
    (temp_dir / "fxsp-rec.yaml").write_text("reconciliation:\n  keys: [id]\n  fields:\n    - name: id\n", encoding="utf-8")

    specs = load_job_specs(
        config_path=config_file,
        repo_root=temp_dir,
        selected_groups=parse_group_selection("intraday"),
    )

    assert [spec.name for spec in specs] == ["fxsp"]
    assert specs[0].groups == ("intraday",)


def test_select_pending_pair_is_source_triggered():
    """A source should only match to a later target, not to an older one."""
    source = SimpleNamespace(
        signature="source-1",
        sort_timestamp=datetime(2026, 3, 27, 10, 0, 0),
    )
    older_target = SimpleNamespace(
        signature="target-old",
        sort_timestamp=datetime(2026, 3, 27, 9, 55, 0),
    )
    later_target = SimpleNamespace(
        signature="target-new",
        sort_timestamp=datetime(2026, 3, 27, 10, 5, 0),
    )
    job = MarketDataJobSpec(
        name="fxsp",
        config_path=Path("fxsp-rec.yaml"),
        output_stem="fxsp",
        source_globs=("fxsp/ProcessedFiles/FXSP_MUREX_EOD-DATA_TODAY_*.csv",),
        target_globs=("fxrate_rep_itd_*.csv",),
        max_target_lag=timedelta(minutes=30),
    )

    pair = select_pending_pair(
        job=job,
        sources=[source],
        targets=[later_target, older_target],
        processed_source_signatures=set(),
        clock_skew_seconds=0,
    )

    assert pair == (source, later_target)


def test_market_data_automation_service_marks_processed_targets_and_skips_rerun(temp_dir):
    """A successfully processed source should not be rerun on the next cycle."""
    source_dir = temp_dir / "source"
    target_dir = temp_dir / "target"
    output_dir = temp_dir / "output"
    source_dir.mkdir()
    target_dir.mkdir()

    source_file = source_dir / "FXSP_MUREX_EOD-DATA_TODAY_1.csv"
    target_file = target_dir / "fxrate_rep_eod_20260327_200704.csv"
    config_file = temp_dir / "fxsp-rec.yaml"

    source_file.write_text("id\n1\n", encoding="utf-8")
    target_file.write_text("id\n1\n", encoding="utf-8")
    config_file.write_text("reconciliation:\n  keys: [id]\n  fields:\n    - name: id\n", encoding="utf-8")

    base_time = datetime(2026, 3, 27, 20, 7, 30).timestamp()
    os.utime(source_file, (base_time - 60, base_time - 60))
    os.utime(target_file, (base_time, base_time))

    calls = []

    class FakeRunner:
        def run(self, request):
            calls.append(request)
            output_path = Path(request.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("xlsx", encoding="utf-8")
            return SimpleNamespace(
                metadata=SimpleNamespace(output_file=str(output_path))
            )

    job = MarketDataJobSpec(
        name="fxsp",
        config_path=config_file,
        output_stem="fxsp",
        source_globs=("FXSP_MUREX_EOD-DATA_TODAY_*.csv",),
        target_globs=("fxrate_rep_*.csv",),
        max_target_lag=timedelta(hours=3),
    )
    state_path = temp_dir / ".market_data_state.json"
    state = AutomationState(state_path)
    state.load()

    service = MarketDataAutomationService(
        job_specs=[job],
        source_dir=source_dir,
        target_dir=target_dir,
        output_dir=output_dir,
        state=state,
        runner=FakeRunner(),
        min_file_age_seconds=0,
    )

    first_summary = service.run_once()
    assert first_summary.processed == 1
    assert first_summary.failed == 0
    assert len(calls) == 1
    assert state_path.exists()

    reloaded_state = AutomationState(state_path)
    reloaded_state.load()
    second_service = MarketDataAutomationService(
        job_specs=[job],
        source_dir=source_dir,
        target_dir=target_dir,
        output_dir=output_dir,
        state=reloaded_state,
        runner=FakeRunner(),
        min_file_age_seconds=0,
    )

    second_summary = second_service.run_once()
    assert second_summary.processed == 0
    assert second_summary.failed == 0


def test_market_data_automation_service_sends_email_when_configured(temp_dir, monkeypatch):
    """Email delivery should be invoked after a successful reconciliation."""
    source_dir = temp_dir / "source"
    target_dir = temp_dir / "target"
    output_dir = temp_dir / "output"
    source_dir.mkdir()
    target_dir.mkdir()

    source_file = source_dir / "RTSH_MUREX_EOD-DATA_TODAY_1.csv"
    target_file = target_dir / "md_ratecurve_rep_eod_20260327_204536.csv"
    config_file = temp_dir / "rtsh-rec.yaml"

    source_file.write_text("id\n1\n", encoding="utf-8")
    target_file.write_text("id\n1\n", encoding="utf-8")
    config_file.write_text("reconciliation:\n  keys: [id]\n  fields:\n    - name: id\n", encoding="utf-8")

    base_time = datetime(2026, 3, 27, 20, 46, 0).timestamp()
    os.utime(source_file, (base_time - 60, base_time - 60))
    os.utime(target_file, (base_time, base_time))

    sent_messages = []

    def fake_send_email(email_settings, attachment_path, job, source, target):
        sent_messages.append(
            {
                "recipients": email_settings.recipients,
                "attachment": attachment_path.name,
                "job": job.name,
                "source": source.path.name,
                "target": target.path.name,
            }
        )

    monkeypatch.setattr("src.market_data_automation.send_email", fake_send_email)

    class FakeRunner:
        def run(self, request):
            output_path = Path(request.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("xlsx", encoding="utf-8")
            return SimpleNamespace(
                metadata=SimpleNamespace(output_file=str(output_path))
            )

    job = MarketDataJobSpec(
        name="rtsh",
        config_path=config_file,
        output_stem="rtsh",
        source_globs=("RTSH_MUREX_EOD-DATA_TODAY_*.csv",),
        target_globs=("md_ratecurve_rep_*.csv",),
    )

    service = MarketDataAutomationService(
        job_specs=[job],
        source_dir=source_dir,
        target_dir=target_dir,
        output_dir=output_dir,
        state=AutomationState(temp_dir / ".market_data_state.json"),
        runner=FakeRunner(),
        email_settings=EmailSettings(
            recipients=("ops@example.com",),
            sender="trex@example.com",
            smtp_host="smtp.example.com",
            smtp_port=25,
        ),
        min_file_age_seconds=0,
    )

    summary = service.run_once()
    assert summary.processed == 1
    assert summary.failed == 0
    assert sent_messages == [
        {
            "recipients": ("ops@example.com",),
            "attachment": "rtsh_20260327_204536.xlsx",
            "job": "rtsh",
            "source": "RTSH_MUREX_EOD-DATA_TODAY_1.csv",
            "target": "md_ratecurve_rep_eod_20260327_204536.csv",
        }
    ]


def test_market_data_automation_service_sends_stuck_alert_once(temp_dir, monkeypatch):
    """An overdue source should trigger one stuck alert and not spam retries."""
    source_dir = temp_dir / "source"
    target_dir = temp_dir / "target"
    output_dir = temp_dir / "output"
    source_dir.mkdir()
    target_dir.mkdir()

    source_file = source_dir / "FXSP_MUREX_EOD-DATA_TODAY_1.csv"
    config_file = temp_dir / "fxsp-rec.yaml"

    source_file.write_text("id\n1\n", encoding="utf-8")
    config_file.write_text("reconciliation:\n  keys: [id]\n  fields:\n    - name: id\n", encoding="utf-8")

    old_time = datetime.now().timestamp() - (31 * 60)
    os.utime(source_file, (old_time, old_time))

    sent_alerts = []

    def fake_send_stuck_alert_email(email_settings, job, source):
        sent_alerts.append(
            {
                "job": job.name,
                "source": source.path.name,
                "recipients": email_settings.recipients,
            }
        )

    monkeypatch.setattr("src.market_data_automation.send_stuck_alert_email", fake_send_stuck_alert_email)

    job = MarketDataJobSpec(
        name="fxsp",
        config_path=config_file,
        output_stem="fxsp",
        source_globs=("FXSP_MUREX_EOD-DATA_TODAY_*.csv",),
        target_globs=("fxrate_rep_*.csv",),
        max_target_lag=timedelta(minutes=30),
    )
    state_path = temp_dir / ".market_data_state.json"
    service = MarketDataAutomationService(
        job_specs=[job],
        source_dir=source_dir,
        target_dir=target_dir,
        output_dir=output_dir,
        state=AutomationState(state_path),
        email_settings=EmailSettings(
            recipients=("ops@example.com",),
            sender="trex@example.com",
            smtp_host="smtp.example.com",
            smtp_port=25,
        ),
        min_file_age_seconds=0,
        clock_skew_seconds=0,
    )

    first_summary = service.run_once()
    assert first_summary.processed == 0
    assert first_summary.failed == 0
    assert first_summary.alerts_sent == 1
    assert sent_alerts == [
        {
            "job": "fxsp",
            "source": "FXSP_MUREX_EOD-DATA_TODAY_1.csv",
            "recipients": ("ops@example.com",),
        }
    ]

    second_summary = service.run_once()
    assert second_summary.processed == 0
    assert second_summary.failed == 0
    assert second_summary.alerts_sent == 0
    assert len(sent_alerts) == 1
