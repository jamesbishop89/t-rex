"""
Tests for shared reconciliation orchestration.
"""

from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.reconciliation_runner import ReconciliationRunRequest, ReconciliationRunner


class TestReconciliationRunner:
    """Test cases for ReconciliationRunner."""

    def _make_work_dir(self) -> Path:
        work_dir = Path(".codex-runner-tests") / uuid4().hex
        work_dir.mkdir(parents=True, exist_ok=False)
        return work_dir

    def test_runner_uses_generated_output_and_returns_metadata(self):
        """Runner should coordinate components and emit metadata in a stable shape."""
        call_log = {}
        work_dir = self._make_work_dir()
        try:
            source_path = work_dir / "source.csv"
            target_path = work_dir / "target.csv"
            config_path = work_dir / "config.yaml"
            source_path.write_text("id\n1\n", encoding="utf-8")
            target_path.write_text("id\n1\n", encoding="utf-8")
            config_path.write_text("reconciliation:\n  keys: [id]\n  fields:\n    - name: amount\n", encoding="utf-8")

            generated_output = work_dir / "reports" / "generated.xlsx"

            class FakeConfigParser:
                def parse_config(self, path):
                    call_log["config_path"] = path
                    return {
                        "reconciliation": {
                            "keys": [{"name": "id", "source": "id", "target": "id", "target_alternatives": []}],
                            "fields": [{"name": "amount", "source": "amount", "target": "amount", "apply_to": "both"}],
                        },
                        "output": {},
                    }

                def get_output_filename_with_timestamp(self, config):
                    call_log["generated_output_called"] = True
                    return str(generated_output)

            class FakeDataLoader:
                def load_data(self, path):
                    call_log.setdefault("loaded_paths", []).append(path)
                    if path == str(source_path):
                        return pd.DataFrame({"id": [1], "amount": [100.0]})
                    return pd.DataFrame({"id": [1], "amount": [100.0]})

            class FakeReconciliationEngine:
                def __init__(self, config):
                    call_log["engine_config"] = config

                def reconcile(self, source_df, target_df):
                    call_log["source_rows"] = len(source_df)
                    call_log["target_rows"] = len(target_df)
                    return {
                        "records": {
                            "matched": pd.DataFrame(),
                            "different": pd.DataFrame(),
                            "missing_in_source": pd.DataFrame(),
                            "missing_in_target": pd.DataFrame(),
                        },
                        "statistics": {
                            "total_source": len(source_df),
                            "total_target": len(target_df),
                            "matched": 1,
                            "different": 0,
                            "missing_in_source": 0,
                            "missing_in_target": 0,
                        },
                        "field_comparison": {},
                        "config": call_log["engine_config"],
                        "transformations": {},
                        "post_merge_adjustments": {},
                        "raw_data": {
                            "source": source_df,
                            "target": target_df,
                        },
                    }

            class FakeExcelGenerator:
                def generate_excel(self, results, output_path, metadata):
                    call_log["excel_output_path"] = output_path
                    call_log["excel_metadata"] = metadata

            runner = ReconciliationRunner(
                config_parser_cls=FakeConfigParser,
                data_loader_cls=FakeDataLoader,
                reconciliation_engine_cls=FakeReconciliationEngine,
                excel_generator_cls=FakeExcelGenerator,
            )

            run_result = runner.run(
                ReconciliationRunRequest(
                    source_path=str(source_path),
                    target_path=str(target_path),
                    config_path=str(config_path),
                )
            )

            assert call_log["generated_output_called"] is True
            assert call_log["loaded_paths"] == [str(source_path), str(target_path)]
            assert call_log["excel_output_path"] == str(generated_output)
            assert Path(call_log["excel_metadata"]["output_file"]) == generated_output.resolve()
            assert call_log["excel_metadata"]["source_columns"] == ["id", "amount"]
            assert run_result.metadata.output_file == str(generated_output.resolve())
            assert run_result.results["statistics"]["matched"] == 1
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def test_runner_honors_explicit_output_override(self):
        """An explicit output path should bypass config-derived output naming."""
        call_log = {}
        work_dir = self._make_work_dir()
        try:
            source_path = work_dir / "source.csv"
            target_path = work_dir / "target.csv"
            config_path = work_dir / "config.yaml"
            output_path = work_dir / "explicit" / "result.xlsx"
            source_path.write_text("id\n1\n", encoding="utf-8")
            target_path.write_text("id\n1\n", encoding="utf-8")
            config_path.write_text("reconciliation:\n  keys: [id]\n  fields:\n    - name: amount\n", encoding="utf-8")

            class FakeConfigParser:
                def parse_config(self, path):
                    return {
                        "reconciliation": {
                            "keys": [{"name": "id", "source": "id", "target": "id", "target_alternatives": []}],
                            "fields": [{"name": "amount", "source": "amount", "target": "amount", "apply_to": "both"}],
                        },
                        "output": {},
                    }

                def get_output_filename_with_timestamp(self, config):
                    raise AssertionError("config-derived output should not be used")

            class FakeDataLoader:
                def load_data(self, path):
                    return pd.DataFrame({"id": [1], "amount": [100.0]})

            class FakeReconciliationEngine:
                def __init__(self, config):
                    pass

                def reconcile(self, source_df, target_df):
                    return {
                        "records": {
                            "matched": pd.DataFrame(),
                            "different": pd.DataFrame(),
                            "missing_in_source": pd.DataFrame(),
                            "missing_in_target": pd.DataFrame(),
                        },
                        "statistics": {
                            "total_source": 1,
                            "total_target": 1,
                            "matched": 1,
                            "different": 0,
                            "missing_in_source": 0,
                            "missing_in_target": 0,
                        },
                        "field_comparison": {},
                        "config": {},
                        "transformations": {},
                        "post_merge_adjustments": {},
                        "raw_data": {
                            "source": source_df,
                            "target": target_df,
                        },
                    }

            class FakeExcelGenerator:
                def generate_excel(self, results, output_path_value, metadata):
                    call_log["output_path"] = output_path_value

            runner = ReconciliationRunner(
                config_parser_cls=FakeConfigParser,
                data_loader_cls=FakeDataLoader,
                reconciliation_engine_cls=FakeReconciliationEngine,
                excel_generator_cls=FakeExcelGenerator,
            )

            run_result = runner.run(
                ReconciliationRunRequest(
                    source_path=str(source_path),
                    target_path=str(target_path),
                    config_path=str(config_path),
                    output_path=str(output_path),
                )
            )

            assert call_log["output_path"] == str(output_path)
            assert run_result.metadata.output_file == str(output_path.resolve())
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
