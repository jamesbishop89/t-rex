"""
Data Loader Module

Handles loading data from various file formats (CSV, Excel, etc.) using pandas.
Provides unified interface for reading different data sources with error handling.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from src.logger_setup import LoggerMixin


class DataLoader(LoggerMixin):
    """
    Data loader for various file formats supported by pandas.
    
    This class provides a unified interface for loading data from different
    file formats including CSV, Excel, JSON, and other pandas-supported formats.
    Includes error handling and data validation.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.supported_extensions = {
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet,
            '.pkl': self._load_pickle,
            '.pickle': self._load_pickle
        }
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a file into a pandas DataFrame.
        
        Automatically detects file format based on extension and uses
        appropriate pandas loader. Provides comprehensive error handling
        and data validation.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to pandas loader
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or data is invalid
            Exception: For other loading errors
        """
        self.logger.info(f"Loading data from: {file_path}")
        
        # Validate file exists
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Get file extension
        extension = file_path_obj.suffix.lower()
        
        if extension not in self.supported_extensions:
            supported = ', '.join(self.supported_extensions.keys())
            raise ValueError(f"Unsupported file format: {extension}. Supported: {supported}")
        
        try:
            # Load data using appropriate loader
            loader_func = self.supported_extensions[extension]
            df = loader_func(file_path, **kwargs)
            
            # Validate loaded data
            self._validate_dataframe(df, file_path)
            
            self.logger.info(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            self.logger.debug(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def _load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file using pandas.read_csv.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            pd.DataFrame: Loaded CSV data
        """
        default_args = {
            'encoding': 'utf-8',
            'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan'],
            'keep_default_na': True
        }
        default_args.update(kwargs)
        
        return pd.read_csv(file_path, **default_args)
    
    def _load_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load Excel file using pandas.read_excel.
        
        Args:
            file_path: Path to Excel file
            **kwargs: Additional arguments for pandas.read_excel
            
        Returns:
            pd.DataFrame: Loaded Excel data
        """
        default_args = {
            'na_values': ['', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan'],
            'keep_default_na': True
        }
        default_args.update(kwargs)
        
        return pd.read_excel(file_path, **default_args)
    
    def _load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load JSON file using pandas.read_json.
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments for pandas.read_json
            
        Returns:
            pd.DataFrame: Loaded JSON data
        """
        return pd.read_json(file_path, **kwargs)
    
    def _load_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load Parquet file using pandas.read_parquet.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments for pandas.read_parquet
            
        Returns:
            pd.DataFrame: Loaded Parquet data
        """
        return pd.read_parquet(file_path, **kwargs)
    
    def _load_pickle(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load Pickle file using pandas.read_pickle.
        
        Args:
            file_path: Path to Pickle file
            **kwargs: Additional arguments for pandas.read_pickle
            
        Returns:
            pd.DataFrame: Loaded Pickle data
        """
        return pd.read_pickle(file_path, **kwargs)
    
    def _validate_dataframe(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Validate loaded DataFrame for basic requirements.
        
        Args:
            df: Loaded DataFrame to validate
            file_path: Path to source file (for error messages)
            
        Raises:
            ValueError: If DataFrame is invalid
        """
        if df is None:
            raise ValueError(f"Failed to load data from {file_path}: DataFrame is None")
        
        if df.empty:
            raise ValueError(f"Data file is empty: {file_path}")
        
        if len(df.columns) == 0:
            raise ValueError(f"Data file has no columns: {file_path}")
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            self.logger.warning(f"Found completely empty columns: {empty_columns}")
        
        # Log data info
        self.logger.debug(f"Data types: {df.dtypes.to_dict()}")
        self.logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict[str, Any]: Information about the DataFrame including
            shape, columns, data types, memory usage, and null counts
        """
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': df.isnull().sum().to_dict(),
            'total_nulls': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
    
    def preprocess_data(self, df: pd.DataFrame, 
                       strip_strings: bool = True,
                       standardize_nulls: bool = True) -> pd.DataFrame:
        """
        Apply common preprocessing steps to DataFrame.
        
        Args:
            df: DataFrame to preprocess
            strip_strings: Whether to strip whitespace from string columns
            standardize_nulls: Whether to standardize null representations
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        self.logger.info("Applying data preprocessing")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        if strip_strings:
            # Strip whitespace from string columns
            string_columns = processed_df.select_dtypes(include=['object']).columns
            for col in string_columns:
                processed_df[col] = processed_df[col].astype(str).str.strip()
                # Convert back to NaN for strings that became empty
                processed_df[col] = processed_df[col].replace('', pd.NA)
        
        if standardize_nulls:
            # Standardize various null representations
            null_values = ['NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan', 'None', 'none', '']
            processed_df = processed_df.replace(null_values, pd.NA)
        
        self.logger.info("Data preprocessing completed")
        return processed_df
