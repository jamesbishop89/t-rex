"""
Test configuration and fixtures for T-Rex unit tests.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'reconciliation': {
            'keys': ['id'],
            'fields': [
                {
                    'name': 'amount',
                    'tolerance': 0.01
                },
                {
                    'name': 'status',
                    'mapping': {'Active': 'A', 'Inactive': 'I'}
                },
                {
                    'name': 'date',
                    'transformation': 'lambda x: x.strip()'
                }
            ]
        }
    }


@pytest.fixture
def sample_source_data():
    """Sample source dataset for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'amount': [100.00, 200.00, 300.00, 400.00],
        'status': ['Active', 'Inactive', 'Active', 'Inactive'],
        'date': ['2025-01-01 ', '2025-01-02', ' 2025-01-03', '2025-01-04 ']
    })


@pytest.fixture
def sample_target_data():
    """Sample target dataset for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 5],
        'amount': [100.00, 200.01, 299.99, 500.00],
        'status': ['A', 'I', 'A', 'A'],
        'date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-05']
    })


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Sample YAML configuration file."""
    config_content = """
reconciliation:
  keys: [id]
  fields:
    - name: amount
      tolerance: 0.01
    - name: status
      mapping:
        Active: A
        Inactive: I
    - name: date
      transformation: "lambda x: x.strip()"
"""
    config_file = temp_dir / "config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def sample_csv_files(temp_dir):
    """Sample CSV files for testing."""
    source_data = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'amount': [100.00, 200.00, 300.00, 400.00],
        'status': ['Active', 'Inactive', 'Active', 'Inactive']
    })
    
    target_data = pd.DataFrame({
        'id': [1, 2, 3, 5],
        'amount': [100.00, 200.01, 299.99, 500.00],
        'status': ['A', 'I', 'A', 'A']
    })
    
    source_file = temp_dir / "source.csv"
    target_file = temp_dir / "target.csv"
    
    source_data.to_csv(source_file, index=False)
    target_data.to_csv(target_file, index=False)
    
    return str(source_file), str(target_file)
