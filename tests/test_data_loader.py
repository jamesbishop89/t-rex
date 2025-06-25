"""
Unit tests for data loader module.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.supported_extensions is not None
        assert '.csv' in loader.supported_extensions
        assert '.xlsx' in loader.supported_extensions
    
    def test_load_csv_file(self, sample_csv_files):
        """Test loading CSV files."""
        source_file, target_file = sample_csv_files
        loader = DataLoader()
        
        df = loader.load_data(source_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'id' in df.columns
        assert 'amount' in df.columns
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_data('nonexistent.csv')
    
    def test_load_unsupported_format(self, temp_dir):
        """Test loading unsupported file format."""
        unsupported_file = temp_dir / "test.txt"
        unsupported_file.write_text("some content")
        
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_data(str(unsupported_file))
    
    def test_load_empty_csv(self, temp_dir):
        """Test loading empty CSV file."""
        empty_csv = temp_dir / "empty.csv"
        empty_csv.write_text("")
        
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="Data file is empty"):
            loader.load_data(str(empty_csv))
    
    def test_validate_dataframe_valid(self):
        """Test validating valid DataFrame."""
        loader = DataLoader()
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        # Should not raise any exception
        loader._validate_dataframe(df, "test.csv")
    
    def test_validate_dataframe_none(self):
        """Test validating None DataFrame."""
        loader = DataLoader()
        
        with pytest.raises(ValueError, match="DataFrame is None"):
            loader._validate_dataframe(None, "test.csv")
    
    def test_validate_dataframe_empty(self):
        """Test validating empty DataFrame."""
        loader = DataLoader()
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Data file is empty"):
            loader._validate_dataframe(df, "test.csv")
    
    def test_validate_dataframe_no_columns(self):
        """Test validating DataFrame with no columns."""
        loader = DataLoader()
        df = pd.DataFrame(index=[0, 1, 2])  # DataFrame with rows but no columns
        
        with pytest.raises(ValueError, match="Data file has no columns"):
            loader._validate_dataframe(df, "test.csv")
    
    def test_get_data_info(self):
        """Test getting DataFrame information."""
        loader = DataLoader()
        df = pd.DataFrame({
            'col1': [1, 2, None],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        
        info = loader.get_data_info(df)
        
        assert info['shape'] == (3, 3)
        assert len(info['columns']) == 3
        assert 'col1' in info['columns']
        assert 'dtypes' in info
        assert info['total_nulls'] == 1
        assert 'memory_usage_mb' in info
    
    def test_preprocess_data_strip_strings(self):
        """Test preprocessing data with string stripping."""
        loader = DataLoader()
        df = pd.DataFrame({
            'col1': ['  value1  ', 'value2 ', ' value3'],
            'col2': [1, 2, 3]
        })
        
        processed_df = loader.preprocess_data(df, strip_strings=True, standardize_nulls=False)
        
        assert processed_df['col1'].iloc[0] == 'value1'
        assert processed_df['col1'].iloc[1] == 'value2'
        assert processed_df['col1'].iloc[2] == 'value3'
    
    def test_preprocess_data_standardize_nulls(self):
        """Test preprocessing data with null standardization."""
        loader = DataLoader()
        df = pd.DataFrame({
            'col1': ['NULL', 'N/A', 'value3'],
            'col2': ['', 'NaN', '3']
        })
        
        processed_df = loader.preprocess_data(df, strip_strings=False, standardize_nulls=True)
        
        assert pd.isna(processed_df['col1'].iloc[0])
        assert pd.isna(processed_df['col1'].iloc[1])
        assert processed_df['col1'].iloc[2] == 'value3'
        assert pd.isna(processed_df['col2'].iloc[0])
        assert pd.isna(processed_df['col2'].iloc[1])
    
    def test_load_csv_with_na_values(self, temp_dir):
        """Test loading CSV with custom NA value handling."""
        csv_content = """id,value,status
1,100,Active
2,NULL,Inactive
3,N/A,Active"""
        
        csv_file = temp_dir / "test_na.csv"
        csv_file.write_text(csv_content)
        
        loader = DataLoader()
        df = loader.load_data(str(csv_file))
        
        # Check that NULL and N/A are treated as NaN
        assert pd.isna(df['value'].iloc[1])
        assert pd.isna(df['status'].iloc[2])
    
    def test_load_excel_file(self, temp_dir):
        """Test loading Excel file (if openpyxl is available)."""
        # Create a simple Excel file
        df_test = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        excel_file = temp_dir / "test.xlsx"
        df_test.to_excel(excel_file, index=False)
        
        loader = DataLoader()
        df_loaded = loader.load_data(str(excel_file))
        
        assert isinstance(df_loaded, pd.DataFrame)
        assert len(df_loaded) == 3
        assert 'id' in df_loaded.columns
        assert 'value' in df_loaded.columns
    
    def test_load_json_file(self, temp_dir):
        """Test loading JSON file."""
        json_content = """[
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 3, "value": 30}
        ]"""
        
        json_file = temp_dir / "test.json"
        json_file.write_text(json_content)
        
        loader = DataLoader()
        df = loader.load_data(str(json_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'id' in df.columns
        assert 'value' in df.columns
