"""
Unit tests for data filtering in ReconciliationEngine.
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.reconciliation_engine import ReconciliationEngine

class TestReconciliationFilters:
    """Test cases for data filtering logic."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'value': [100, 200, 300, 400, 500],
            'status': ['Active', 'Inactive', 'Active', 'Pending', 'Active']
        })
    
    def test_filter_equals(self, sample_data):
        """Test 'equals' filter."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [{'name': 'value'}],
                'filters': {
                    'source': [
                        {'field': 'category', 'condition': 'equals', 'value': 'A'}
                    ]
                }
            }
        }
        
        engine = ReconciliationEngine(config)
        filtered_df = engine._apply_filters(sample_data, 'source')
        
        assert len(filtered_df) == 2
        assert all(filtered_df['category'] == 'A')
        assert list(filtered_df['id']) == [1, 3]
        
    def test_filter_not_equals(self, sample_data):
        """Test 'not_equals' filter."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [{'name': 'value'}],
                'filters': {
                    'source': [
                        {'field': 'category', 'condition': 'not_equals', 'value': 'A'}
                    ]
                }
            }
        }
        
        engine = ReconciliationEngine(config)
        filtered_df = engine._apply_filters(sample_data, 'source')
        
        assert len(filtered_df) == 3
        assert all(filtered_df['category'] != 'A')
        
    def test_filter_greater_than(self, sample_data):
        """Test 'greater_than' filter."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [{'name': 'value'}],
                'filters': {
                    'source': [
                        {'field': 'value', 'condition': 'greater_than', 'value': 300}
                    ]
                }
            }
        }
        
        engine = ReconciliationEngine(config)
        filtered_df = engine._apply_filters(sample_data, 'source')
        
        assert len(filtered_df) == 2
        assert all(filtered_df['value'] > 300)
        assert list(filtered_df['id']) == [4, 5]
        
    def test_filter_in_list(self, sample_data):
        """Test 'in_list' filter."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [{'name': 'value'}],
                'filters': {
                    'source': [
                        {'field': 'category', 'condition': 'in_list', 'values': ['A', 'C']}
                    ]
                }
            }
        }
        
        engine = ReconciliationEngine(config)
        filtered_df = engine._apply_filters(sample_data, 'source')
        
        assert len(filtered_df) == 3
        assert all(filtered_df['category'].isin(['A', 'C']))
        
    def test_multiple_filters(self, sample_data):
        """Test applying multiple filters."""
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [{'name': 'value'}],
                'filters': {
                    'source': [
                        {'field': 'category', 'condition': 'equals', 'value': 'A'},
                        {'field': 'value', 'condition': 'greater_than', 'value': 100}
                    ]
                }
            }
        }
        
        engine = ReconciliationEngine(config)
        filtered_df = engine._apply_filters(sample_data, 'source')
        
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]['id'] == 3
        
    def test_filter_missing_field(self, sample_data):
        """Test filter on missing field (should be ignored or return empty depending on logic)."""
        # Current logic logs warning and returns False mask (empty result)
        config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [{'name': 'value'}],
                'filters': {
                    'source': [
                        {'field': 'non_existent', 'condition': 'equals', 'value': 'A'}
                    ]
                }
            }
        }
        
        engine = ReconciliationEngine(config)
        filtered_df = engine._apply_filters(sample_data, 'source')
        
        assert len(filtered_df) == 0
