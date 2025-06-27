"""
Unit tests for reconciliation engine module.
"""

import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.reconciliation_engine import ReconciliationEngine


class TestReconciliationEngine:
    """Test cases for ReconciliationEngine class."""
    
    def test_init(self, sample_config):
        """Test ReconciliationEngine initialization."""
        engine = ReconciliationEngine(sample_config)
        
        assert engine.config == sample_config
        assert engine.keys == ['id']
        assert len(engine.fields) == 3
        assert len(engine.field_configs) == 3
    
    def test_reconcile_basic(self, sample_config, sample_source_data, sample_target_data):
        """Test basic reconciliation process."""
        engine = ReconciliationEngine(sample_config)
        results = engine.reconcile(sample_source_data, sample_target_data)
        
        assert 'records' in results
        assert 'statistics' in results
        assert 'field_comparison' in results
        assert 'config' in results
        
        # Check record categories
        assert 'matched' in results['records']
        assert 'different' in results['records']
        assert 'missing_in_source' in results['records']
        assert 'missing_in_target' in results['records']
    
    def test_validate_input_data_valid(self, sample_config, sample_source_data, sample_target_data):
        """Test input data validation with valid data."""
        engine = ReconciliationEngine(sample_config)
        
        # Should not raise any exception
        engine._validate_input_data(sample_source_data, sample_target_data)
    
    def test_validate_input_data_missing_key_source(self, sample_config, sample_target_data):
        """Test input data validation with missing key in source."""
        engine = ReconciliationEngine(sample_config)
        
        # Create source data without 'id' column
        source_data = pd.DataFrame({
            'amount': [100, 200],
            'status': ['Active', 'Inactive']
        })
        
        with pytest.raises(ValueError, match="Missing reconciliation keys in source data"):
            engine._validate_input_data(source_data, sample_target_data)
    
    def test_validate_input_data_missing_key_target(self, sample_config, sample_source_data):
        """Test input data validation with missing key in target."""
        engine = ReconciliationEngine(sample_config)
        
        # Create target data without 'id' column
        target_data = pd.DataFrame({
            'amount': [100, 200],
            'status': ['A', 'I']
        })
        
        with pytest.raises(ValueError, match="Missing reconciliation keys in target data"):
            engine._validate_input_data(sample_source_data, target_data)
    
    def test_apply_mapping(self, sample_config):
        """Test applying field mappings."""
        engine = ReconciliationEngine(sample_config)
        
        df = pd.DataFrame({
            'status': ['Active', 'Inactive', 'Active', 'Unknown']
        })
        
        mapping = {'Active': 'A', 'Inactive': 'I'}
        result_df = engine._apply_mapping(df, 'status', mapping, 'test')
        
        assert result_df['status'].iloc[0] == 'A'
        assert result_df['status'].iloc[1] == 'I'
        assert result_df['status'].iloc[2] == 'A'
        assert result_df['status'].iloc[3] == 'Unknown'  # Unmapped value remains
    
    def test_apply_transformation(self, sample_config):
        """Test applying field transformations."""
        engine = ReconciliationEngine(sample_config)
        
        df = pd.DataFrame({
            'date': ['  2025-01-01  ', '2025-01-02 ', ' 2025-01-03', None]
        })
        
        transformation = 'lambda x: x.strip()'
        result_df = engine._apply_transformation(df, 'date', transformation, 'test')
        
        assert result_df['date'].iloc[0] == '2025-01-01'
        assert result_df['date'].iloc[1] == '2025-01-02'
        assert result_df['date'].iloc[2] == '2025-01-03'
        assert pd.isna(result_df['date'].iloc[3])  # None remains None
    
    def test_apply_transformation_invalid(self, sample_config):
        """Test applying invalid transformation."""
        engine = ReconciliationEngine(sample_config)
        
        df = pd.DataFrame({'field': ['value1', 'value2']})
        
        with pytest.raises(ValueError, match="Transformation failed"):
            engine._apply_transformation(df, 'field', 'invalid lambda', 'test')
    
    def test_compare_with_tolerance_absolute(self, sample_config):
        """Test tolerance comparison with absolute values."""
        engine = ReconciliationEngine(sample_config)
        
        source_series = pd.Series([100.00, 200.00, 300.00])
        target_series = pd.Series([100.01, 199.99, 301.00])
        
        # Test with 0.01 tolerance
        matches = engine._compare_with_tolerance(source_series, target_series, 0.01)
        assert matches.iloc[0] == True   # 0.01 difference, within tolerance
        assert matches.iloc[1] == True   # 0.01 difference, within tolerance
        assert matches.iloc[2] == False  # 1.00 difference, outside tolerance
        
        # Test with 1.0 tolerance
        matches = engine._compare_with_tolerance(source_series, target_series, 1.0)
        assert matches.iloc[0] == True   # All within 1.0 tolerance
        assert matches.iloc[1] == True
        assert matches.iloc[2] == True
    
    def test_compare_with_tolerance_percentage(self, sample_config):
        """Test tolerance comparison with percentage values."""
        engine = ReconciliationEngine(sample_config)
        
        source_series = pd.Series([100.00, 200.00, 0.00])
        target_series = pd.Series([101.00, 198.00, 0.00])
        
        # Test with 1% tolerance
        matches = engine._compare_with_tolerance(source_series, target_series, "1%")
        assert matches.iloc[0] == True   # 1% difference, within tolerance
        assert matches.iloc[1] == True   # 1% difference, within tolerance
        assert matches.iloc[2] == True   # Both zero, should match
        
        # Test with 0.5% tolerance
        matches = engine._compare_with_tolerance(source_series, target_series, "0.5%")
        assert matches.iloc[0] == False  # 1% difference, outside 0.5% tolerance
        assert matches.iloc[1] == False  # 1% difference, outside 0.5% tolerance
        assert matches.iloc[2] == True   # Both zero, should match
    
    def test_compare_with_tolerance_nan_values(self, sample_config):
        """Test tolerance comparison with NaN values."""
        engine = ReconciliationEngine(sample_config)
        
        source_series = pd.Series([100.00, np.nan, 300.00])
        target_series = pd.Series([100.01, np.nan, np.nan])
        
        matches = engine._compare_with_tolerance(source_series, target_series, 0.01)
        assert matches.iloc[0] == True   # Within tolerance
        assert matches.iloc[1] == True   # Both NaN, should match
        assert matches.iloc[2] == False  # One NaN, one value, should not match
    
    def test_merge_datasets(self, sample_config, sample_source_data, sample_target_data):
        """Test merging source and target datasets."""
        engine = ReconciliationEngine(sample_config)
        
        # Preprocess the data first
        source_processed = engine._preprocess_data(sample_source_data.copy(), 'source')
        target_processed = engine._preprocess_data(sample_target_data.copy(), 'target')
        
        merged_df = engine._merge_datasets(source_processed, target_processed)
        
        assert '_merge_indicator' in merged_df.columns
        assert len(merged_df) >= max(len(sample_source_data), len(sample_target_data))
        
        # Check that we have source_ and target_ prefixed columns
        source_cols = [col for col in merged_df.columns if col.startswith('source_')]
        target_cols = [col for col in merged_df.columns if col.startswith('target_')]
        
        assert len(source_cols) > 0
        assert len(target_cols) > 0
    
    def test_categorize_records(self, sample_config, sample_source_data, sample_target_data):
        """Test categorizing records into different categories."""
        engine = ReconciliationEngine(sample_config)
        
        # Preprocess and merge data
        source_processed = engine._preprocess_data(sample_source_data.copy(), 'source')
        target_processed = engine._preprocess_data(sample_target_data.copy(), 'target')
        merged_df = engine._merge_datasets(source_processed, target_processed)
        
        # Perform field comparison
        comparison_results = engine._compare_fields(merged_df)
        
        # Categorize records
        categorized = engine._categorize_records(merged_df, comparison_results)
        
        assert 'matched' in categorized
        assert 'different' in categorized
        assert 'missing_in_source' in categorized
        assert 'missing_in_target' in categorized
        
        # Check that all categorized records are DataFrames
        for category, df in categorized.items():
            assert isinstance(df, pd.DataFrame)
    
    def test_calculate_statistics(self, sample_config, sample_source_data, sample_target_data):
        """Test calculating reconciliation statistics."""
        engine = ReconciliationEngine(sample_config)
        
        # Create mock categorized records
        categorized_records = {
            'matched': pd.DataFrame({'id': [1, 2]}),
            'different': pd.DataFrame({'id': [3]}),
            'missing_in_source': pd.DataFrame({'id': [5]}),
            'missing_in_target': pd.DataFrame({'id': [4]})
        }
        
        # Create mock comparison results
        comparison_results = {
            'amount': {
                'matches_count': 2,
                'total_comparable': 3,
                'match_rate': 0.67
            }
        }
        
        stats = engine._calculate_statistics(
            sample_source_data, sample_target_data, categorized_records, comparison_results
        )
        
        assert stats['total_source'] == len(sample_source_data)
        assert stats['total_target'] == len(sample_target_data)
        assert stats['matched'] == 2
        assert stats['different'] == 1
        assert stats['missing_in_source'] == 1
        assert stats['missing_in_target'] == 1
        assert 'field_statistics' in stats
        assert 'match_rate' in stats
        assert 'difference_rate' in stats
    
    def test_preprocess_data_order(self, sample_config):
        """Test that preprocessing applies mapping before transformation."""
        # Create config with both mapping and transformation
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'status',
                        'mapping': {'Active': 'A', 'Inactive': 'I'},
                        'transformation': 'lambda x: x.lower()'
                    }
                ]
            }
        }
        
        engine = ReconciliationEngine(config)
        
        df = pd.DataFrame({
            'id': [1, 2],
            'status': ['Active', 'Inactive']
        })
        
        processed_df = engine._preprocess_data(df, 'test')
        
        # Mapping should be applied first (Active -> A), then transformation (A -> a)
        assert processed_df['status'].iloc[0] == 'a'
        assert processed_df['status'].iloc[1] == 'i'
    
    def test_field_comparison_no_tolerance(self, sample_config):
        """Test field comparison without tolerance (exact match)."""
        engine = ReconciliationEngine(sample_config)
        
        source_series = pd.Series(['A', 'B', 'C'])
        target_series = pd.Series(['A', 'X', 'C'])
        
        result = engine._compare_field_values(source_series, target_series, None, 'test_field')
        
        assert result['matches'].iloc[0] == True   # A == A
        assert result['matches'].iloc[1] == False  # B != X
        assert result['matches'].iloc[2] == True   # C == C
        assert result['matches_count'] == 2
        assert result['total_comparable'] == 3
        assert result['match_rate'] == 2/3

    def test_ignore_field_functionality(self):
        """Test that fields marked as ignore are handled correctly."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'amount',
                        'tolerance': 0.01
                    },
                    {
                        'name': 'status'
                    },
                    {
                        'name': 'comments',
                        'ignore': True
                    },                    {
                        'name': 'timestamp',
                        'ignore': True
                    }
                ]
            }
        }
        
        engine = ReconciliationEngine(config)
        
        # Test field separation
        comparison_field_names = [field['name'] for field in engine.comparison_fields]
        ignored_field_names = [field['name'] for field in engine.ignored_fields]
        
        assert 'amount' in comparison_field_names
        assert 'status' in comparison_field_names
        assert 'comments' in ignored_field_names
        assert 'timestamp' in ignored_field_names
        assert len(engine.comparison_fields) == 2
        assert len(engine.ignored_fields) == 2
        
        # Test reconciliation with ignored fields
        source_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [100.00, 200.00, 300.00],
            'status': ['Active', 'Inactive', 'Active'],
            'comments': ['Comment 1', 'Comment 2', 'Comment 3'],
            'timestamp': ['2025-01-01', '2025-01-02', '2025-01-03']
        })
        
        target_data = pd.DataFrame({
            'id': [1, 2, 3],
            'amount': [100.00, 200.00, 300.00],
            'status': ['Active', 'Inactive', 'Active'],
            'comments': ['Different Comment 1', 'Different Comment 2', 'Different Comment 3'],
            'timestamp': ['2025-02-01', '2025-02-02', '2025-02-03']
        })
        
        results = engine.reconcile(source_data, target_data)
        
        # All records should match since ignored fields don't affect comparison
        assert len(results['records']['matched']) == 3
        assert len(results['records']['different']) == 0
        
        # Field comparison should only include non-ignored fields
        field_results = results['field_comparison']
        assert 'amount' in field_results
        assert 'status' in field_results
        assert 'comments' not in field_results
        assert 'timestamp' not in field_results

    def test_ignore_field_with_different_comparison_fields(self):
        """Test reconciliation where comparison fields differ but ignored fields are present."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'amount',
                        'tolerance': 0.01
                    },
                    {
                        'name': 'notes',
                        'ignore': True
                    }
                ]
            }
        }
        
        engine = ReconciliationEngine(config)
        
        source_data = pd.DataFrame({
            'id': [1, 2],
            'amount': [100.00, 200.00],
            'notes': ['Note 1', 'Note 2']
        })
        
        target_data = pd.DataFrame({
            'id': [1, 2],
            'amount': [100.01, 199.99],  # Within tolerance
            'notes': ['Different Note 1', 'Different Note 2']  # Different but ignored
        })
        
        results = engine.reconcile(source_data, target_data)
        
        # Records should match despite different notes (ignored field)
        assert len(results['records']['matched']) == 2
        assert len(results['records']['different']) == 0

    def test_conditional_mapping(self):
        """Test conditional mapping functionality."""
        # Create test configuration with conditional mapping
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'type',
                            'mappings': {
                                'EQUITY': {
                                    'N': 'New',
                                    'F': 'Filled',
                                    'C': 'Cancelled'
                                },
                                'BOND': {
                                    'N': 'New Order',
                                    'F': 'Complete',
                                    'C': 'Void'
                                }
                            }
                        }
                    },
                    {
                        'name': 'type'  # Condition field must also be configured
                    }
                ]
            }
        }
        
        # Create test data
        source_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'type': ['EQUITY', 'EQUITY', 'BOND', 'BOND'],
            'status': ['N', 'F', 'N', 'C']
        })
        
        target_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'type': ['EQUITY', 'EQUITY', 'BOND', 'BOND'],
            'status': ['New', 'Filled', 'New Order', 'Void']
        })
        
        # Run reconciliation
        engine = ReconciliationEngine(config)
        results = engine.reconcile(source_data, target_data)
        
        # All records should match after conditional mapping
        assert results['statistics']['matched'] == 4
        assert results['statistics']['different'] == 0
        
        # Check that the mapping was applied correctly in the processed data
        matched_records = results['records']['matched']
        assert len(matched_records) == 4
        
        # Verify source values were mapped correctly
        source_status_values = matched_records['source_status'].tolist()
        expected_mapped_values = ['New', 'Filled', 'New Order', 'Void']
        assert source_status_values == expected_mapped_values

    def test_conditional_mapping_with_partial_matches(self):
        """Test conditional mapping where some values don't match conditions."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'type',
                            'mappings': {
                                'EQUITY': {
                                    'N': 'New',
                                    'F': 'Filled'
                                }
                                # No mapping for BOND type
                            }
                        }
                    },
                    {
                        'name': 'type'
                    }
                ]
            }
        }
        
        source_data = pd.DataFrame({
            'id': [1, 2, 3],
            'type': ['EQUITY', 'BOND', 'EQUITY'],
            'status': ['N', 'X', 'F']  # 'X' for BOND type has no mapping
        })
        
        target_data = pd.DataFrame({
            'id': [1, 2, 3],
            'type': ['EQUITY', 'BOND', 'EQUITY'],
            'status': ['New', 'X', 'Filled']
        })
        
        engine = ReconciliationEngine(config)
        results = engine.reconcile(source_data, target_data)
        
        # Should have matches where mapping applied and original values for unmapped
        assert results['statistics']['matched'] == 3
        assert results['statistics']['different'] == 0

    def test_conditional_mapping_missing_condition_field(self):
        """Test conditional mapping when condition field is missing from data."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'missing_field',
                            'mappings': {
                                'EQUITY': {'N': 'New'}
                            }
                        }
                    }
                ]
            }
        }
        
        source_data = pd.DataFrame({
            'id': [1],
            'status': ['N']
            # missing_field is not present
        })
        
        target_data = pd.DataFrame({
            'id': [1],
            'status': ['N']
        })
        
        engine = ReconciliationEngine(config)
        
        # Should handle missing condition field gracefully
        results = engine.reconcile(source_data, target_data)
        
        # Should still work, just without applying conditional mapping
        assert results['statistics']['matched'] == 1

    def test_conditional_mapping_not_starts_with(self):
        """Test conditional mapping with 'not_starts_with' condition type."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'counterparty'
                    },
                    {
                        'name': 'currency1',
                        'conditional_mapping': {
                            'condition_field': 'counterparty',
                            'condition_type': 'not_starts_with',
                            'condition_value': 'ABC',
                            'mappings': {
                                'default': {
                                    'KRA': 'KRW'
                                }
                            }
                        }
                    }
                ]
            }
        }
        
        source_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'counterparty': ['XYZ Corp', 'ABC_Bank', 'DEF Ltd', 'ABC Corp'],
            'currency1': ['KRA', 'KRA', 'KRA', 'KRA']
        })
        
        target_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'counterparty': ['XYZ Corp', 'ABC_Bank', 'DEF Ltd', 'ABC Corp'],
            'currency1': ['KRW', 'KRA', 'KRW', 'KRA']  # Expected results after mapping
        })
        
        engine = ReconciliationEngine(config)
        results = engine.reconcile(source_data, target_data)
        
        # All records should match after conditional mapping is applied
        assert results['statistics']['matched'] == 4
        assert results['statistics']['different'] == 0
        
        # Verify the mapping was applied correctly by checking the processed source data
        # We can access the actual processed data through the reconciliation engine's internal data
        # For testing purposes, we'll verify by running the reconciliation and checking it succeeds
        # The key test is that all records match, which means the conditional mapping worked
