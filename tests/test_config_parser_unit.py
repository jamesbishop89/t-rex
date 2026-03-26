"""
Focused unit tests for config parser behaviors that do not require temp files.
"""

from pathlib import Path

import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_normalizer import normalize_runtime_config
from src.config_parser import ConfigParser


class TestConfigParserUnit:
    """Unit-level parser tests for schema/runtime alignment."""

    def test_schema_accepts_integer_tolerance_and_numeric_condition_value(self):
        """Integer tolerances and numeric condition operands should validate cleanly."""
        parser = ConfigParser()
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'quantity',
                        'tolerance': 1,
                    },
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'quantity',
                            'condition_type': 'greater_than',
                            'condition_value': 0,
                            'mappings': {
                                'default': {'N': 'POSITIVE'}
                            }
                        }
                    }
                ]
            }
        }

        validated = parser.config_schema.validate(config)
        normalized = normalize_runtime_config(validated)

        parser._validate_conditional_mappings(normalized)

        fields = normalized['reconciliation']['fields']
        assert fields[0]['tolerance'] == 1
        assert fields[1]['conditional_mapping']['condition_value'] == 0

    def test_validate_conditional_mappings_allows_false_condition_value(self):
        """Boolean false is a valid condition operand and should not be treated as missing."""
        parser = ConfigParser()
        config = normalize_runtime_config({
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {'name': 'is_internal'},
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'is_internal',
                            'condition_type': 'equals',
                            'condition_value': False,
                            'mappings': {
                                'default': {'N': 'EXTERNAL'}
                            }
                        }
                    }
                ]
            }
        })

        parser._validate_conditional_mappings(config)

    def test_validate_conditional_mappings_rejects_missing_required_operand(self):
        """Normalized configs with value-based conditions still require a real operand."""
        parser = ConfigParser()
        config = normalize_runtime_config({
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {'name': 'quantity'},
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'quantity',
                            'condition_type': 'greater_than',
                            'mappings': {
                                'default': {'N': 'POSITIVE'}
                            }
                        }
                    }
                ]
            }
        })

        with pytest.raises(ValueError, match="missing required 'condition_value'"):
            parser._validate_conditional_mappings(config)

    def test_parse_tolerance_rejects_boolean_values(self):
        """Booleans should not be accepted as numeric tolerances."""
        parser = ConfigParser()

        with pytest.raises(ValueError, match="Invalid tolerance format"):
            parser.parse_tolerance(True)

    def test_parse_shipped_listed_option_config(self):
        """The shipped listed option config should parse under the safe lambda rules."""
        parser = ConfigParser()
        repo_root = Path(__file__).resolve().parents[1]

        config = parser.parse_config(
            str(repo_root / 'config' / 'trade-attributes' / 'listedOption-rec.yaml')
        )

        assert config['reconciliation']['fields']
