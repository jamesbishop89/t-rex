"""
Integration tests for T-Rex reconciliation tool.
"""

import pytest
import pandas as pd
import tempfile
import yaml
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_parser import ConfigParser
from src.data_loader import DataLoader
from src.reconciliation_engine import ReconciliationEngine
from src.excel_generator import ExcelGenerator


class TestIntegration:
    """Integration tests for complete T-Rex workflow."""
    
    def test_end_to_end_reconciliation(self, temp_dir):
        """Test complete end-to-end reconciliation process."""
        # Create test data files
        source_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'amount': [100.00, 200.00, 300.00, 400.00, 500.00],
            'status': ['Active', 'Inactive', 'Active', 'Inactive', 'Active'],
            'date': ['2025-01-01 ', '2025-01-02', ' 2025-01-03', '2025-01-04 ', '2025-01-05']
        })
        
        target_data = pd.DataFrame({
            'id': [1, 2, 3, 6, 7],
            'amount': [100.01, 199.99, 300.00, 600.00, 700.00],
            'status': ['A', 'I', 'A', 'A', 'I'],
            'date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07']
        })
        
        # Save data files
        source_file = temp_dir / "source.csv"
        target_file = temp_dir / "target.csv"
        source_data.to_csv(source_file, index=False)
        target_data.to_csv(target_file, index=False)
        
        # Create configuration file
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'amount',
                        'tolerance': 0.02
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
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Step 1: Parse configuration
        config_parser = ConfigParser()
        parsed_config = config_parser.parse_config(str(config_file))
        
        # Step 2: Load data
        data_loader = DataLoader()
        source_df = data_loader.load_data(str(source_file))
        target_df = data_loader.load_data(str(target_file))
        
        # Step 3: Run reconciliation
        reconciliation_engine = ReconciliationEngine(parsed_config)
        results = reconciliation_engine.reconcile(source_df, target_df)
        
        # Step 4: Generate Excel output
        output_file = temp_dir / "output.xlsx"
        metadata = {
            'source_file': str(source_file),
            'target_file': str(target_file),
            'config_file': str(config_file),
            'execution_time': 1.0,
            'recon_date': '2025-06-24'
        }
        
        excel_generator = ExcelGenerator()
        excel_generator.generate_excel(results, str(output_file), metadata)
        
        # Verify results
        assert output_file.exists()
        
        # Check statistics
        stats = results['statistics']
        assert stats['total_source'] == 5
        assert stats['total_target'] == 5
        assert stats['matched'] == 3  # Records 1, 2, 3 should match
        assert stats['different'] == 0  # No differences within tolerance
        assert stats['missing_in_source'] == 2  # Records 6, 7
        assert stats['missing_in_target'] == 2  # Records 4, 5
        
        # Check record categories
        records = results['records']
        assert len(records['matched']) == 3
        assert len(records['different']) == 0
        assert len(records['missing_in_source']) == 2
        assert len(records['missing_in_target']) == 2
    
    def test_reconciliation_with_differences(self, temp_dir):
        """Test reconciliation with records that have differences."""
        # Create test data with clear differences
        source_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [100.00, 200.00, 300.00],
            'status': ['Active', 'Inactive', 'Active']
        })
        
        target_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [100.00, 250.00, 300.00],  # Record 2 has significant difference
            'status': ['A', 'A', 'A']  # Record 2 status is different after mapping
        })
        
        # Save data files
        source_file = temp_dir / "source.csv"
        target_file = temp_dir / "target.csv"
        source_data.to_csv(source_file, index=False)
        target_data.to_csv(target_file, index=False)
        
        # Create configuration with tight tolerance
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'amount',
                        'tolerance': 1.0  # 50 difference is outside tolerance
                    },
                    {
                        'name': 'status',
                        'mapping': {'Active': 'A', 'Inactive': 'I'}
                    }
                ]
            }
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Run reconciliation
        config_parser = ConfigParser()
        parsed_config = config_parser.parse_config(str(config_file))
        
        data_loader = DataLoader()
        source_df = data_loader.load_data(str(source_file))
        target_df = data_loader.load_data(str(target_file))
        
        reconciliation_engine = ReconciliationEngine(parsed_config)
        results = reconciliation_engine.reconcile(source_df, target_df)
        
        # Check that record 2 is classified as different
        stats = results['statistics']
        assert stats['matched'] == 2  # Records 1 and 3
        assert stats['different'] == 1  # Record 2
        assert stats['missing_in_source'] == 0
        assert stats['missing_in_target'] == 0
    
    def test_percentage_tolerance(self, temp_dir):
        """Test reconciliation with percentage-based tolerance."""
        source_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [100.00, 200.00, 1000.00]
        })
        
        target_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [101.00, 204.00, 1050.00]  # 1%, 2%, 5% differences
        })
        
        # Save data files
        source_file = temp_dir / "source.csv"
        target_file = temp_dir / "target.csv"
        source_data.to_csv(source_file, index=False)
        target_data.to_csv(target_file, index=False)
        
        # Create configuration with 3% tolerance
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'amount',
                        'tolerance': '3%'
                    }
                ]
            }
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Run reconciliation
        config_parser = ConfigParser()
        parsed_config = config_parser.parse_config(str(config_file))
        
        data_loader = DataLoader()
        source_df = data_loader.load_data(str(source_file))
        target_df = data_loader.load_data(str(target_file))
        
        reconciliation_engine = ReconciliationEngine(parsed_config)
        results = reconciliation_engine.reconcile(source_df, target_df)
        
        # Check results
        stats = results['statistics']
        assert stats['matched'] == 2  # Records 1 and 2 within 3% tolerance
        assert stats['different'] == 1  # Record 3 outside 3% tolerance
    
    def test_missing_fields_handling(self, temp_dir):
        """Test handling of missing fields in source or target data."""
        # Source has extra field
        source_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [100.00, 200.00, 300.00],
            'extra_field': ['A', 'B', 'C']
        })
        
        # Target missing extra_field
        target_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [100.00, 200.00, 300.00]
        })
        
        # Save data files
        source_file = temp_dir / "source.csv"
        target_file = temp_dir / "target.csv"
        source_data.to_csv(source_file, index=False)
        target_data.to_csv(target_file, index=False)
        
        # Configure field that exists in both
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {'name': 'amount'},
                    {'name': 'extra_field'}  # This field missing in target
                ]
            }
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Run reconciliation
        config_parser = ConfigParser()
        parsed_config = config_parser.parse_config(str(config_file))
        
        data_loader = DataLoader()
        source_df = data_loader.load_data(str(source_file))
        target_df = data_loader.load_data(str(target_file))
        
        reconciliation_engine = ReconciliationEngine(parsed_config)
        
        # Should complete without error, handling missing field gracefully
        results = reconciliation_engine.reconcile(source_df, target_df)
        
        # Should still have results based on available fields
        assert 'statistics' in results
        assert results['statistics']['total_source'] == 3
        assert results['statistics']['total_target'] == 3
