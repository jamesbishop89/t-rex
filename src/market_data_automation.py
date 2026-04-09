"""
Production automation for market-data reconciliations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
import fnmatch
from html import escape
import json
import logging
import os
from pathlib import Path
import re
import smtplib
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import yaml

from src.logger_setup import setup_logger
from src.reconciliation_runner import ReconciliationRunRequest, ReconciliationRunner


TIMESTAMP_PATTERN = re.compile(r"(?P<stamp>\d{8}_\d{6})")
DEFAULT_AUTOMATION_CONFIG = "config/market-data/automation.yaml"
DEFAULT_EMAIL_SUBJECT = "T-Rex market data reconciliation: {job_name}"
DEFAULT_EMAIL_BODY = (
    "Market data reconciliation completed.\n\n"
    "Job: {job_name}\n"
    "Source: {source_path}\n"
    "Target: {target_path}\n"
    "Output: {output_path}\n"
    "Config: {config_path}\n"
)
DEFAULT_EMAIL_HTML_BODY = """\
<html>
  <body style="margin:0;padding:24px;background:#f4f7fb;font-family:Segoe UI,Arial,sans-serif;color:#1f2937;">
    <div style="max-width:720px;margin:0 auto;background:#ffffff;border:1px solid #dbe4f0;border-radius:12px;overflow:hidden;">
      <div style="padding:20px 24px;background:#0f172a;color:#ffffff;">
        <div style="font-size:12px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.8;">T-Rex</div>
        <div style="font-size:22px;font-weight:700;margin-top:6px;">Market Data Reconciliation Complete</div>
      </div>
      <div style="padding:24px;">
        <p style="margin:0 0 16px 0;font-size:14px;line-height:1.6;">
          The reconciliation workbook has been generated successfully.
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
          <tr>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;width:170px;font-weight:600;">Job</td>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;">{job_name}</td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;font-weight:600;">Source File</td>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;"><code>{source_name}</code></td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;font-weight:600;">Target File</td>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;"><code>{target_name}</code></td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;font-weight:600;">Workbook</td>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;"><code>{output_name}</code></td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;font-weight:600;vertical-align:top;">Source Path</td>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;"><code>{source_path}</code></td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;font-weight:600;vertical-align:top;">Target Path</td>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;"><code>{target_path}</code></td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;font-weight:600;vertical-align:top;">Output Path</td>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;"><code>{output_path}</code></td>
          </tr>
          <tr>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;border-bottom:1px solid #e5e7eb;font-weight:600;vertical-align:top;">Config Path</td>
            <td style="padding:10px 0;border-top:1px solid #e5e7eb;border-bottom:1px solid #e5e7eb;"><code>{config_path}</code></td>
          </tr>
        </table>
      </div>
    </div>
  </body>
</html>
"""
DEFAULT_STUCK_EMAIL_SUBJECT = "T-Rex market data reconciliation stuck: {job_name}"
DEFAULT_STUCK_EMAIL_BODY = (
    "Market data reconciliation is waiting for a target file longer than allowed.\n\n"
    "Job: {job_name}\n"
    "Source: {source_path}\n"
    "Source seen at: {source_seen_at}\n"
    "Expected target pattern(s): {target_patterns}\n"
    "Max target lag minutes: {max_target_lag_minutes}\n"
)


@dataclass(frozen=True)
class MarketDataJobSpec:
    """One market-data reconciliation job definition."""

    name: str
    config_path: Path
    output_stem: str
    source_globs: tuple[str, ...]
    target_globs: tuple[str, ...]
    groups: tuple[str, ...] = ()
    source_exclude_globs: tuple[str, ...] = ()
    target_exclude_globs: tuple[str, ...] = ()
    max_target_lag: timedelta = timedelta(hours=3)


@dataclass(frozen=True)
class FileCandidate:
    """A stable source/target file candidate."""

    path: Path
    sort_timestamp: datetime
    modified_timestamp: datetime
    size_bytes: int
    modified_timestamp_ns: int

    @property
    def signature(self) -> str:
        """Return a stable signature used to avoid duplicate processing."""
        return f"{self.path}|{self.size_bytes}|{self.modified_timestamp_ns}"


@dataclass(frozen=True)
class RunCycleSummary:
    """Aggregate outcome for one polling cycle."""

    processed: int = 0
    failed: int = 0
    alerts_sent: int = 0


@dataclass(frozen=True)
class EmailSettings:
    """SMTP delivery settings for reconciliation outputs."""

    recipients: tuple[str, ...]
    sender: str
    smtp_host: str
    smtp_port: int
    smtp_username: str = ""
    smtp_password: str = ""
    use_starttls: bool = False
    use_ssl: bool = False
    subject_template: str = DEFAULT_EMAIL_SUBJECT
    body_template: str = DEFAULT_EMAIL_BODY
    html_body_template: str = DEFAULT_EMAIL_HTML_BODY

    @property
    def enabled(self) -> bool:
        return bool(self.recipients)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for market-data automation."""
    parser = argparse.ArgumentParser(
        description=(
            "Poll source and target folders for market-data reconciliations, "
            "run T-Rex when a fresh target arrives, and optionally email the workbook."
        )
    )
    parser.add_argument(
        "--automation-config",
        default=DEFAULT_AUTOMATION_CONFIG,
        help=f"YAML job map (default: {DEFAULT_AUTOMATION_CONFIG}).",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory where market-data source files arrive.",
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Directory where market-data target files arrive.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Excel outputs should be written.",
    )
    parser.add_argument(
        "--state-file",
        default="",
        help="Optional JSON state file path. Defaults to <output-dir>/.market_data_state.json.",
    )
    parser.add_argument(
        "--jobs",
        default="",
        help="Comma-separated subset of jobs to run. Default is all enabled jobs.",
    )
    parser.add_argument(
        "--job-groups",
        default="",
        help="Comma-separated job groups to run, for example 'intraday' or 'eod'. Default is all groups.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=60,
        help="Polling interval when running continuously (default: 60).",
    )
    parser.add_argument(
        "--min-file-age-seconds",
        type=int,
        default=30,
        help="Only consider files older than this many seconds (default: 30).",
    )
    parser.add_argument(
        "--clock-skew-seconds",
        type=int,
        default=60,
        help="Allow the source to appear slightly after the target by this many seconds (default: 60).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one polling cycle and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would run without generating or emailing anything.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for the automation loop.",
    )
    parser.add_argument(
        "--email-to",
        default=os.environ.get("TREX_EMAIL_TO", ""),
        help="Comma-separated recipient list. Defaults to TREX_EMAIL_TO.",
    )
    parser.add_argument(
        "--email-from",
        default=os.environ.get("TREX_EMAIL_FROM", ""),
        help="Sender address. Defaults to TREX_EMAIL_FROM.",
    )
    parser.add_argument(
        "--smtp-host",
        default=os.environ.get("TREX_SMTP_HOST", ""),
        help="SMTP host. Defaults to TREX_SMTP_HOST.",
    )
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=int(os.environ.get("TREX_SMTP_PORT", "25")),
        help="SMTP port. Defaults to TREX_SMTP_PORT or 25.",
    )
    parser.add_argument(
        "--smtp-username",
        default=os.environ.get("TREX_SMTP_USERNAME", ""),
        help="SMTP username. Defaults to TREX_SMTP_USERNAME.",
    )
    parser.add_argument(
        "--smtp-password",
        default=os.environ.get("TREX_SMTP_PASSWORD", ""),
        help="SMTP password. Defaults to TREX_SMTP_PASSWORD.",
    )
    parser.add_argument(
        "--smtp-starttls",
        action="store_true",
        help="Enable SMTP STARTTLS.",
    )
    parser.add_argument(
        "--smtp-ssl",
        action="store_true",
        help="Use SMTP over SSL.",
    )
    parser.add_argument(
        "--email-subject",
        default=DEFAULT_EMAIL_SUBJECT,
        help="Email subject template.",
    )
    parser.add_argument(
        "--email-body",
        default=DEFAULT_EMAIL_BODY,
        help="Email body template.",
    )
    parser.add_argument(
        "--email-html-body",
        default=DEFAULT_EMAIL_HTML_BODY,
        help="HTML email body template.",
    )
    return parser.parse_args()


def parse_timestamp_from_name(file_name: str) -> Optional[datetime]:
    """Extract a YYYYMMDD_HHMMSS timestamp from a file name when present."""
    matches = list(TIMESTAMP_PATTERN.finditer(file_name))
    if not matches:
        return None
    return datetime.strptime(matches[-1].group("stamp"), "%Y%m%d_%H%M%S")


def resolve_reference(path_value: str, base_dir: Path, repo_root: Path) -> Path:
    """Resolve a config reference relative to the config file and repo root."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    for anchor in (base_dir, repo_root):
        candidate = (anchor / path).resolve()
        if candidate.exists():
            return candidate

    return (base_dir / path).resolve()


def load_job_specs(
    config_path: Path,
    repo_root: Path,
    selected_jobs: Optional[Set[str]] = None,
    selected_groups: Optional[Set[str]] = None,
) -> List[MarketDataJobSpec]:
    """Load automation job specs from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Automation config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    jobs_section = raw_config.get("jobs")
    if not isinstance(jobs_section, list) or not jobs_section:
        raise ValueError("Automation config must define a non-empty 'jobs' list")

    specs: List[MarketDataJobSpec] = []
    config_dir = config_path.parent
    for entry in jobs_section:
        if not isinstance(entry, dict):
            raise ValueError("Each job entry must be a mapping")
        if entry.get("enabled", True) is False:
            continue

        name = str(entry["name"]).strip()
        if selected_jobs and name not in selected_jobs:
            continue

        groups = tuple(str(value).strip() for value in entry.get("groups", []) if str(value).strip())
        if selected_groups and not selected_groups.intersection(groups):
            continue

        source_globs = tuple(str(value) for value in entry.get("source_globs", []))
        target_globs = tuple(str(value) for value in entry.get("target_globs", []))
        if not source_globs or not target_globs:
            raise ValueError(f"Job '{name}' must define source_globs and target_globs")

        specs.append(
            MarketDataJobSpec(
                name=name,
                config_path=resolve_reference(str(entry["config"]), config_dir, repo_root),
                output_stem=str(entry.get("output_stem", name)).strip() or name,
                source_globs=source_globs,
                target_globs=target_globs,
                groups=groups,
                source_exclude_globs=tuple(str(value) for value in entry.get("source_exclude_globs", [])),
                target_exclude_globs=tuple(str(value) for value in entry.get("target_exclude_globs", [])),
                max_target_lag=timedelta(minutes=int(entry.get("max_target_lag_minutes", 180))),
            )
        )

    if selected_jobs:
        loaded_names = {spec.name for spec in specs}
        missing = sorted(selected_jobs - loaded_names)
        if missing:
            raise ValueError(f"Unknown or disabled job(s): {', '.join(missing)}")

    return specs


def should_exclude(path: Path, exclude_globs: Sequence[str]) -> bool:
    """Return True when the candidate matches one of the exclude globs."""
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in exclude_globs)


def collect_file_candidates(
    directory: Path,
    include_globs: Sequence[str],
    exclude_globs: Sequence[str],
    min_file_age_seconds: int,
    now: Optional[datetime] = None,
) -> List[FileCandidate]:
    """Collect stable file candidates from a directory."""
    if not directory.exists():
        return []

    observed_at = now or datetime.now()
    candidates_by_path: Dict[Path, FileCandidate] = {}
    for pattern in include_globs:
        for path in directory.glob(pattern):
            if not path.is_file() or should_exclude(path, exclude_globs):
                continue
            stat = path.stat()
            modified = datetime.fromtimestamp(stat.st_mtime)
            age_seconds = (observed_at - modified).total_seconds()
            if age_seconds < min_file_age_seconds:
                continue

            sort_timestamp = parse_timestamp_from_name(path.name) or modified
            candidates_by_path[path.resolve()] = FileCandidate(
                path=path.resolve(),
                sort_timestamp=sort_timestamp,
                modified_timestamp=modified,
                size_bytes=stat.st_size,
                modified_timestamp_ns=stat.st_mtime_ns,
            )

    return sorted(
        candidates_by_path.values(),
        key=lambda candidate: (candidate.sort_timestamp, candidate.modified_timestamp, candidate.path.name),
        reverse=True,
    )


def select_matching_target(
    targets: Sequence[FileCandidate],
    source: FileCandidate,
    max_target_lag: timedelta,
    clock_skew_seconds: int,
) -> Optional[FileCandidate]:
    """Pick the newest target that plausibly belongs with the source."""
    for target in targets:
        lag = target.sort_timestamp - source.sort_timestamp
        if lag.total_seconds() < -clock_skew_seconds:
            continue
        if lag > max_target_lag:
            continue
        return target
    return None


def select_pending_pair(
    job: MarketDataJobSpec,
    sources: Sequence[FileCandidate],
    targets: Sequence[FileCandidate],
    processed_source_signatures: Set[str],
    clock_skew_seconds: int,
) -> Optional[tuple[FileCandidate, FileCandidate]]:
    """Choose the next source-triggered source/target pair that should be reconciled."""
    for source in reversed(sources):
        if source.signature in processed_source_signatures:
            continue
        target = select_matching_target(
            targets=targets,
            source=source,
            max_target_lag=job.max_target_lag,
            clock_skew_seconds=clock_skew_seconds,
        )
        if target is not None:
            return source, target
    return None


class AutomationState:
    """Persistent JSON state for already-processed source-triggered runs."""

    def __init__(self, path: Path):
        self.path = path
        self._state: Dict[str, Any] = {"jobs": {}}

    def load(self) -> None:
        """Load existing state from disk when present."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            self._state = json.load(handle)

    def save(self) -> None:
        """Persist state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2, sort_keys=True)

    def processed_sources_for(self, job_name: str) -> Set[str]:
        """Return processed source signatures for the job."""
        job_state = self._state.get("jobs", {}).get(job_name, {})
        processed = job_state.get("processed_runs", job_state.get("processed_targets", []))
        return {
            str(item["source_signature"])
            for item in processed
            if isinstance(item, dict) and "source_signature" in item
        }

    def alerted_sources_for(self, job_name: str) -> Set[str]:
        """Return source signatures that already triggered a stuck alert."""
        job_state = self._state.get("jobs", {}).get(job_name, {})
        alerts = job_state.get("stuck_alerts", [])
        return {
            str(item["source_signature"])
            for item in alerts
            if isinstance(item, dict) and "source_signature" in item
        }

    def mark_processed(
        self,
        job_name: str,
        source: FileCandidate,
        target: FileCandidate,
        output_path: Path,
    ) -> None:
        """Record a successful target delivery."""
        jobs_state = self._state.setdefault("jobs", {})
        job_state = jobs_state.setdefault(job_name, {})
        processed = job_state.setdefault("processed_runs", [])
        processed.append(
            {
                "processed_at": datetime.now().isoformat(timespec="seconds"),
                "source_path": str(source.path),
                "source_signature": source.signature,
                "target_path": str(target.path),
                "target_signature": target.signature,
                "output_path": str(output_path),
            }
        )
        if len(processed) > 50:
            del processed[:-50]

    def mark_stuck_alerted(
        self,
        job_name: str,
        source: FileCandidate,
    ) -> None:
        """Record that a stuck alert has already been sent for this source."""
        jobs_state = self._state.setdefault("jobs", {})
        job_state = jobs_state.setdefault(job_name, {})
        alerts = job_state.setdefault("stuck_alerts", [])
        alerts.append(
            {
                "alerted_at": datetime.now().isoformat(timespec="seconds"),
                "source_path": str(source.path),
                "source_signature": source.signature,
            }
        )
        if len(alerts) > 50:
            del alerts[:-50]


def build_email_settings(args: argparse.Namespace) -> Optional[EmailSettings]:
    """Create SMTP settings when email delivery is configured."""
    recipients = tuple(
        value.strip()
        for value in args.email_to.split(",")
        if value.strip()
    )
    if not recipients:
        return None

    required = {
        "email-from": args.email_from,
        "smtp-host": args.smtp_host,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Email delivery is enabled but missing required settings: "
            + ", ".join(missing)
        )

    return EmailSettings(
        recipients=recipients,
        sender=args.email_from,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_username=args.smtp_username,
        smtp_password=args.smtp_password,
        use_starttls=args.smtp_starttls,
        use_ssl=args.smtp_ssl,
        subject_template=args.email_subject,
        body_template=args.email_body,
        html_body_template=args.email_html_body,
    )


def build_reconciliation_email_message(
    email_settings: EmailSettings,
    attachment_path: Path,
    job: MarketDataJobSpec,
    source: FileCandidate,
    target: FileCandidate,
) -> EmailMessage:
    """Build the reconciliation result email with plain-text and HTML bodies."""
    message = EmailMessage()
    format_kwargs = {
        "job_name": job.name,
        "source_name": source.path.name,
        "source_path": str(source.path),
        "target_name": target.path.name,
        "target_path": str(target.path),
        "output_name": attachment_path.name,
        "output_path": str(attachment_path),
        "config_path": str(job.config_path),
    }
    html_format_kwargs = {
        key: escape(value) if isinstance(value, str) else value
        for key, value in format_kwargs.items()
    }
    message["Subject"] = email_settings.subject_template.format(**format_kwargs)
    message["From"] = email_settings.sender
    message["To"] = ", ".join(email_settings.recipients)
    message.set_content(email_settings.body_template.format(**format_kwargs))
    message.add_alternative(
        email_settings.html_body_template.format(**html_format_kwargs),
        subtype="html",
    )

    with open(attachment_path, "rb") as workbook:
        message.add_attachment(
            workbook.read(),
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=attachment_path.name,
        )
    return message


def send_email(
    email_settings: EmailSettings,
    attachment_path: Path,
    job: MarketDataJobSpec,
    source: FileCandidate,
    target: FileCandidate,
) -> None:
    """Send the generated reconciliation workbook."""
    message = build_reconciliation_email_message(
        email_settings=email_settings,
        attachment_path=attachment_path,
        job=job,
        source=source,
        target=target,
    )

    smtp_factory = smtplib.SMTP_SSL if email_settings.use_ssl else smtplib.SMTP
    with smtp_factory(email_settings.smtp_host, email_settings.smtp_port, timeout=30) as client:
        if email_settings.use_starttls and not email_settings.use_ssl:
            client.starttls()
        if email_settings.smtp_username:
            client.login(email_settings.smtp_username, email_settings.smtp_password)
        client.send_message(message)


def send_stuck_alert_email(
    email_settings: EmailSettings,
    job: MarketDataJobSpec,
    source: FileCandidate,
) -> None:
    """Send a stuck-source alert when the target does not arrive in time."""
    message = EmailMessage()
    format_kwargs = {
        "job_name": job.name,
        "source_name": source.path.name,
        "source_path": str(source.path),
        "source_seen_at": source.sort_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "target_patterns": ", ".join(job.target_globs),
        "max_target_lag_minutes": int(job.max_target_lag.total_seconds() // 60),
    }
    message["Subject"] = DEFAULT_STUCK_EMAIL_SUBJECT.format(**format_kwargs)
    message["From"] = email_settings.sender
    message["To"] = ", ".join(email_settings.recipients)
    message.set_content(DEFAULT_STUCK_EMAIL_BODY.format(**format_kwargs))

    smtp_factory = smtplib.SMTP_SSL if email_settings.use_ssl else smtplib.SMTP
    with smtp_factory(email_settings.smtp_host, email_settings.smtp_port, timeout=30) as client:
        if email_settings.use_starttls and not email_settings.use_ssl:
            client.starttls()
        if email_settings.smtp_username:
            client.login(email_settings.smtp_username, email_settings.smtp_password)
        client.send_message(message)


class MarketDataAutomationService:
    """Polling service for market-data reconciliation automation."""

    def __init__(
        self,
        job_specs: Sequence[MarketDataJobSpec],
        source_dir: Path,
        target_dir: Path,
        output_dir: Path,
        state: AutomationState,
        runner: Optional[ReconciliationRunner] = None,
        email_settings: Optional[EmailSettings] = None,
        min_file_age_seconds: int = 30,
        clock_skew_seconds: int = 60,
    ):
        self.job_specs = list(job_specs)
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.state = state
        self.runner = runner or ReconciliationRunner()
        self.email_settings = email_settings
        self.min_file_age_seconds = min_file_age_seconds
        self.clock_skew_seconds = clock_skew_seconds
        self.logger = logging.getLogger(__name__)

    def build_output_path(self, job: MarketDataJobSpec, target: FileCandidate) -> Path:
        """Build a deterministic output path for a target file."""
        timestamp = target.sort_timestamp.strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{job.output_stem}_{timestamp}.xlsx"

    def _find_overdue_sources(
        self,
        job: MarketDataJobSpec,
        sources: Sequence[FileCandidate],
        targets: Sequence[FileCandidate],
        processed_source_signatures: Set[str],
        alerted_source_signatures: Set[str],
        observed_at: datetime,
    ) -> List[FileCandidate]:
        """Return sources that have exceeded the allowed wait for a target."""
        overdue_sources: List[FileCandidate] = []
        for source in sources:
            if source.signature in processed_source_signatures:
                continue
            if source.signature in alerted_source_signatures:
                continue
            if select_matching_target(
                targets=targets,
                source=source,
                max_target_lag=job.max_target_lag,
                clock_skew_seconds=self.clock_skew_seconds,
            ) is not None:
                continue
            if observed_at - source.sort_timestamp > job.max_target_lag:
                overdue_sources.append(source)
        return overdue_sources

    def run_once(self, dry_run: bool = False) -> RunCycleSummary:
        """Run one polling cycle across all configured jobs."""
        processed_count = 0
        failed_count = 0
        alerts_sent = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for job in self.job_specs:
            try:
                observed_at = datetime.now()
                sources = collect_file_candidates(
                    directory=self.source_dir,
                    include_globs=job.source_globs,
                    exclude_globs=job.source_exclude_globs,
                    min_file_age_seconds=self.min_file_age_seconds,
                    now=observed_at,
                )
                targets = collect_file_candidates(
                    directory=self.target_dir,
                    include_globs=job.target_globs,
                    exclude_globs=job.target_exclude_globs,
                    min_file_age_seconds=self.min_file_age_seconds,
                    now=observed_at,
                )
                processed_source_signatures = self.state.processed_sources_for(job.name)
                alerted_source_signatures = self.state.alerted_sources_for(job.name)
                overdue_sources = self._find_overdue_sources(
                    job=job,
                    sources=sources,
                    targets=targets,
                    processed_source_signatures=processed_source_signatures,
                    alerted_source_signatures=alerted_source_signatures,
                    observed_at=observed_at,
                )
                for source in overdue_sources:
                    self.logger.warning(
                        "Job '%s' source '%s' exceeded max target lag of %s minutes",
                        job.name,
                        source.path.name,
                        int(job.max_target_lag.total_seconds() // 60),
                    )
                    if dry_run:
                        self.logger.info("Dry run: would send stuck alert for %s", source.path)
                        alerts_sent += 1
                        continue
                    if self.email_settings is None or not self.email_settings.enabled:
                        self.logger.warning(
                            "Email settings are not configured, cannot send stuck alert for job '%s'",
                            job.name,
                        )
                        continue
                    send_stuck_alert_email(
                        email_settings=self.email_settings,
                        job=job,
                        source=source,
                    )
                    self.state.mark_stuck_alerted(job.name, source)
                    self.state.save()
                    alerts_sent += 1
                pair = select_pending_pair(
                    job=job,
                    sources=sources,
                    targets=targets,
                    processed_source_signatures=processed_source_signatures,
                    clock_skew_seconds=self.clock_skew_seconds,
                )
                if pair is None:
                    self.logger.debug("No pending pair for job '%s'", job.name)
                    continue

                source, target = pair
                output_path = self.build_output_path(job, target)
                self.logger.info(
                    "Job '%s' matched source '%s' with target '%s'",
                    job.name,
                    source.path.name,
                    target.path.name,
                )

                if dry_run:
                    self.logger.info("Dry run: would generate %s", output_path)
                    processed_count += 1
                    continue

                run_result = self.runner.run(
                    ReconciliationRunRequest(
                        source_path=str(source.path),
                        target_path=str(target.path),
                        config_path=str(job.config_path),
                        output_path=str(output_path),
                    )
                )
                workbook_path = Path(run_result.metadata.output_file)

                if self.email_settings is not None and self.email_settings.enabled:
                    send_email(
                        email_settings=self.email_settings,
                        attachment_path=workbook_path,
                        job=job,
                        source=source,
                        target=target,
                    )
                    self.logger.info(
                        "Emailed workbook for job '%s' to %s",
                        job.name,
                        ", ".join(self.email_settings.recipients),
                    )

                self.state.mark_processed(job.name, source, target, workbook_path)
                self.state.save()
                processed_count += 1
            except Exception:
                failed_count += 1
                self.logger.exception("Job '%s' failed", job.name)

        return RunCycleSummary(
            processed=processed_count,
            failed=failed_count,
            alerts_sent=alerts_sent,
        )


def parse_job_selection(jobs_arg: str) -> Optional[Set[str]]:
    """Parse the --jobs argument."""
    names = {name.strip() for name in jobs_arg.split(",") if name.strip()}
    return names or None


def parse_group_selection(groups_arg: str) -> Optional[Set[str]]:
    """Parse the --job-groups argument."""
    groups = {name.strip() for name in groups_arg.split(",") if name.strip()}
    return groups or None


def default_state_path(output_dir: Path) -> Path:
    """Return the default state file path for an output folder."""
    return output_dir / ".market_data_state.json"


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    setup_logger(args.log_level)

    repo_root = Path(__file__).resolve().parents[1]
    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    state_path = Path(args.state_file).resolve() if args.state_file else default_state_path(output_dir)
    selected_jobs = parse_job_selection(args.jobs)
    selected_groups = parse_group_selection(args.job_groups)
    email_settings = build_email_settings(args)
    job_specs = load_job_specs(
        config_path=Path(args.automation_config).resolve(),
        repo_root=repo_root,
        selected_jobs=selected_jobs,
        selected_groups=selected_groups,
    )

    state = AutomationState(state_path)
    state.load()
    service = MarketDataAutomationService(
        job_specs=job_specs,
        source_dir=source_dir,
        target_dir=target_dir,
        output_dir=output_dir,
        state=state,
        email_settings=email_settings,
        min_file_age_seconds=args.min_file_age_seconds,
        clock_skew_seconds=args.clock_skew_seconds,
    )

    logger = logging.getLogger(__name__)
    logger.info("Loaded %s market-data job(s)", len(job_specs))
    logger.info("Source dir: %s", source_dir)
    logger.info("Target dir: %s", target_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("State file: %s", state_path)
    if selected_groups:
        logger.info("Job groups: %s", ", ".join(sorted(selected_groups)))
    if selected_jobs:
        logger.info("Jobs: %s", ", ".join(sorted(selected_jobs)))

    if args.once:
        summary = service.run_once(dry_run=args.dry_run)
        return 1 if summary.failed else 0

    while True:
        try:
            summary = service.run_once(dry_run=args.dry_run)
            if summary.failed:
                logger.warning("Cycle finished with %s failed job(s)", summary.failed)
            time.sleep(args.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Market-data automation stopped by user")
            return 0
        except Exception:
            logger.exception("Market-data automation cycle failed")
            return 1


if __name__ == "__main__":
    sys.exit(main())
