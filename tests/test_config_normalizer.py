"""
Tests for runtime config normalization helpers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_normalizer import normalize_runtime_config


class TestConfigNormalizer:
    """Test cases for runtime config normalization."""

    def test_normalize_runtime_config_defaults(self, sample_config):
        """Normalization should make runtime defaults explicit."""
        normalized = normalize_runtime_config(sample_config)

        key_config = normalized["reconciliation"]["keys"][0]
        amount_field = normalized["reconciliation"]["fields"][0]

        assert key_config["name"] == "id"
        assert key_config["source"] == "id"
        assert key_config["target"] == "id"
        assert key_config["target_alternatives"] == []

        assert amount_field["name"] == "amount"
        assert amount_field["source"] == "amount"
        assert amount_field["target"] == "amount"
        assert amount_field["apply_to"] == "both"
        assert amount_field["ignore"] is False
        assert normalized["output"] == {}
