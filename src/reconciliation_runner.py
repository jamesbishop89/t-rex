"""
Shared orchestration service for reconciliation runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Type

from src.config_parser import ConfigParser
from src.data_loader import DataLoader
from src.excel_generator import ExcelGenerator
from src.reconciliation_engine import ReconciliationEngine


@dataclass(frozen=True)
class ReconciliationRunRequest:
    """Inputs required to execute a reconciliation run."""

    source_path: str
    target_path: str
    config_path: str
    output_path: Optional[str] = None


@dataclass(frozen=True)
class ReconciliationRunMetadata:
    """Execution metadata emitted for a reconciliation run."""

    source_file: str
    target_file: str
    config_file: str
    output_file: str
    execution_time: float
    recon_date: str
    source_columns: list[str]

    def as_dict(self) -> Dict[str, Any]:
        """Return metadata as a plain dictionary for existing consumers."""
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationRunResult:
    """Structured result returned by the runner service."""

    results: Dict[str, Any]
    metadata: ReconciliationRunMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Flatten into the legacy dict shape used by the CLI."""
        return {**self.results, "metadata": self.metadata.as_dict()}


class ReconciliationRunner:
    """Coordinate config parsing, data loading, reconciliation, and reporting."""

    def __init__(
        self,
        config_parser_cls: Type[ConfigParser] = ConfigParser,
        data_loader_cls: Type[DataLoader] = DataLoader,
        reconciliation_engine_cls: Type[ReconciliationEngine] = ReconciliationEngine,
        excel_generator_cls: Type[ExcelGenerator] = ExcelGenerator,
    ):
        self._config_parser_cls = config_parser_cls
        self._data_loader_cls = data_loader_cls
        self._reconciliation_engine_cls = reconciliation_engine_cls
        self._excel_generator_cls = excel_generator_cls

    @staticmethod
    def validate_input_paths(request: ReconciliationRunRequest) -> None:
        """Validate that the requested input files exist."""
        for label, path_str in (
            ("Source", request.source_path),
            ("Target", request.target_path),
            ("Config", request.config_path),
        ):
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path_str}")

    @staticmethod
    def validate_and_prepare_output_path(output_path: str) -> None:
        """Ensure the output directory exists."""
        output_path_obj = Path(output_path)
        output_dir = output_path_obj.parent

        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise PermissionError(f"Cannot create output directory: {exc}") from exc

        if not output_dir.is_dir():
            raise PermissionError(f"Output directory is not a directory: {output_dir}")

    def run(self, request: ReconciliationRunRequest) -> ReconciliationRunResult:
        """Execute a full reconciliation run."""
        logger = logging.getLogger(__name__)
        start_time = time.time()

        self.validate_input_paths(request)

        logger.info("Starting T-Rex reconciliation process")
        logger.info(f"Source file: {request.source_path}")
        logger.info(f"Target file: {request.target_path}")
        logger.info(f"Config file: {request.config_path}")

        logger.info("Step 1: Parsing configuration file")
        config_parser = self._config_parser_cls()
        config = config_parser.parse_config(request.config_path)
        field_count = len(config["reconciliation"].get("fields", []))
        logger.info(f"Configuration loaded successfully: {field_count} fields configured")

        if request.output_path:
            output_file = request.output_path
            logger.info(f"Using output file from request: {output_file}")
        else:
            output_file = config_parser.get_output_filename_with_timestamp(config)
            logger.info(f"Generated output filename from config: {output_file}")

        self.validate_and_prepare_output_path(output_file)
        logger.info(f"Output file: {output_file}")

        logger.info("Step 2: Loading data files")
        data_loader = self._data_loader_cls()
        source_df = data_loader.load_data(request.source_path)
        target_df = data_loader.load_data(request.target_path)

        logger.info(
            f"Source data loaded: {len(source_df)} rows, {len(source_df.columns)} columns"
        )
        logger.info(
            f"Target data loaded: {len(target_df)} rows, {len(target_df.columns)} columns"
        )

        logger.info("Step 3: Executing reconciliation")
        reconciliation_engine = self._reconciliation_engine_cls(config)
        results = reconciliation_engine.reconcile(source_df, target_df)

        stats = results["statistics"]
        logger.info("Reconciliation completed:")
        logger.info(f"  - Matched records: {stats['matched']}")
        logger.info(f"  - Different records: {stats['different']}")
        logger.info(f"  - Missing in source: {stats['missing_in_source']}")
        logger.info(f"  - Missing in target: {stats['missing_in_target']}")

        logger.info("Step 4: Generating Excel output")
        execution_time = time.time() - start_time
        metadata = ReconciliationRunMetadata(
            source_file=str(Path(request.source_path).absolute()),
            target_file=str(Path(request.target_path).absolute()),
            config_file=str(Path(request.config_path).absolute()),
            output_file=str(Path(output_file).absolute()),
            execution_time=execution_time,
            recon_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            source_columns=list(source_df.columns),
        )

        excel_generator = self._excel_generator_cls()
        excel_generator.generate_excel(results, output_file, metadata.as_dict())
        logger.info(f"Excel output generated: {output_file}")
        logger.info(
            f"T-Rex reconciliation completed successfully in {execution_time:.2f} seconds"
        )

        return ReconciliationRunResult(results=results, metadata=metadata)
