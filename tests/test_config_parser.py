"""
Unit tests for configuration parser module.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from schema import SchemaError

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_parser import ConfigParser


class TestConfigParser:
    """Test cases for ConfigParser class."""
    
    def test_init(self):
        """Test ConfigParser initialization."""
        parser = ConfigParser()
        assert parser.config_schema is not None
    
    def test_parse_valid_config(self, sample_yaml_config):
        """Test parsing valid configuration file."""
        parser = ConfigParser()
        config = parser.parse_config(sample_yaml_config)
        
        assert 'reconciliation' in config
        assert 'keys' in config['reconciliation']
        assert 'fields' in config['reconciliation']
        assert len(config['reconciliation']['keys']) > 0
        assert len(config['reconciliation']['fields']) > 0
    
    def test_parse_nonexistent_file(self):
        """Test parsing non-existent configuration file."""
        parser = ConfigParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_config('nonexistent.yaml')
    
    def test_parse_empty_config(self, temp_dir):
        """Test parsing empty configuration file."""
        config_file = temp_dir / "empty.yaml"
        config_file.write_text("")
        
        parser = ConfigParser()
        
        with pytest.raises(ValueError, match="Configuration file is empty"):
            parser.parse_config(str(config_file))
    
    def test_parse_invalid_yaml(self, temp_dir):
        """Test parsing invalid YAML syntax."""
        config_file = temp_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content:")
        
        parser = ConfigParser()
        
        with pytest.raises(yaml.YAMLError):
            parser.parse_config(str(config_file))
    
    def test_parse_missing_reconciliation_section(self, temp_dir):
        """Test parsing config without reconciliation section."""
        config_content = """
        other_section:
          key: value
        """
        config_file = temp_dir / "no_recon.yaml"
        config_file.write_text(config_content)
        
        parser = ConfigParser()
        
        with pytest.raises(SchemaError):
            parser.parse_config(str(config_file))
    
    def test_parse_missing_keys(self, temp_dir):
        """Test parsing config without reconciliation keys."""
        config_content = """
        reconciliation:
          fields:
            - name: amount
        """
        config_file = temp_dir / "no_keys.yaml"
        config_file.write_text(config_content)
        
        parser = ConfigParser()
        
        with pytest.raises(SchemaError):
            parser.parse_config(str(config_file))
    
    def test_parse_empty_keys(self, temp_dir):
        """Test parsing config with empty keys list."""
        config_content = """
        reconciliation:
          keys: []
          fields:
            - name: amount
        """
        config_file = temp_dir / "empty_keys.yaml"
        config_file.write_text(config_content)
        
        parser = ConfigParser()
        
        with pytest.raises(SchemaError):
            parser.parse_config(str(config_file))
    
    def test_parse_duplicate_field_names(self, temp_dir):
        """Test parsing config with duplicate field names."""
        config_content = """
        reconciliation:
          keys: [id]
          fields:
            - name: amount
            - name: amount
        """
        config_file = temp_dir / "duplicate_fields.yaml"
        config_file.write_text(config_content)
        
        parser = ConfigParser()
        
        with pytest.raises(ValueError, match="Duplicate field names"):
            parser.parse_config(str(config_file))
    
    def test_validate_lambda_valid(self):
        """Test validating valid lambda expressions."""
        parser = ConfigParser()
        
        # Valid lambda expressions
        assert parser._validate_lambda("lambda x: x.strip()")
        assert parser._validate_lambda("lambda x: float(x)")
        assert parser._validate_lambda("lambda x: x.upper()")
    
    def test_validate_lambda_invalid(self):
        """Test validating invalid lambda expressions."""
        parser = ConfigParser()
        
        # Invalid lambda expressions
        with pytest.raises(ValueError, match="must start with 'lambda'"):
            parser._validate_lambda("x.strip()")
        
        with pytest.raises(ValueError, match="Invalid lambda syntax"):
            parser._validate_lambda("lambda x x.strip()")
    
    def test_validate_percentage_tolerance_valid(self):
        """Test validating valid percentage tolerances."""
        parser = ConfigParser()
        
        assert parser._validate_percentage_tolerance("1%")
        assert parser._validate_percentage_tolerance("0.5%")
        assert parser._validate_percentage_tolerance("10%")
        assert parser._validate_percentage_tolerance("0%")
    
    def test_validate_percentage_tolerance_invalid(self):
        """Test validating invalid percentage tolerances."""
        parser = ConfigParser()
        
        with pytest.raises(ValueError, match="must end with '%'"):
            parser._validate_percentage_tolerance("1")
        
        with pytest.raises(ValueError, match="Invalid percentage format"):
            parser._validate_percentage_tolerance("abc%")
        
        with pytest.raises(ValueError, match="must be non-negative"):
            parser._validate_percentage_tolerance("-1%")
    
    def test_get_field_config(self, sample_config):
        """Test getting field configuration."""
        parser = ConfigParser()
        
        # Test existing field
        field_config = parser.get_field_config(sample_config, 'amount')
        assert field_config is not None
        assert field_config['name'] == 'amount'
        assert field_config['tolerance'] == 0.01
        
        # Test non-existing field
        field_config = parser.get_field_config(sample_config, 'nonexistent')
        assert field_config is None
    
    def test_parse_tolerance_absolute(self):
        """Test parsing absolute tolerance values."""
        parser = ConfigParser()
        
        result = parser.parse_tolerance(0.01)
        assert result['type'] == 'absolute'
        assert result['value'] == 0.01
        
        result = parser.parse_tolerance(1)
        assert result['type'] == 'absolute'
        assert result['value'] == 1.0
    
    def test_parse_tolerance_percentage(self):
        """Test parsing percentage tolerance values."""
        parser = ConfigParser()
        
        result = parser.parse_tolerance("1%")
        assert result['type'] == 'percentage'
        assert result['value'] == 0.01
        
        result = parser.parse_tolerance("10%")
        assert result['type'] == 'percentage'
        assert result['value'] == 0.10
    
    def test_parse_tolerance_invalid(self):
        """Test parsing invalid tolerance values."""
        parser = ConfigParser()
        
        with pytest.raises(ValueError, match="Invalid tolerance format"):
            parser.parse_tolerance("invalid")
    
    def test_validate_lambda_function(self):
        """Test validating lambda functions with test execution."""
        parser = ConfigParser()
        
        # Valid lambda that works
        assert parser.validate_lambda_function("lambda x: x.strip()", "  test  ")
        
        # Valid lambda that fails execution
        with pytest.raises(ValueError, match="Lambda function validation failed"):
            parser.validate_lambda_function("lambda x: x.invalid_method()", "test")
    
    def test_conditional_mapping_validation(self, temp_dir):
        """Test validation of conditional mapping configuration."""
        parser = ConfigParser()
        
        # Valid conditional mapping config
        valid_config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'type',
                            'mappings': {
                                'EQUITY': {'N': 'New', 'F': 'Filled'},
                                'BOND': {'N': 'New Order', 'F': 'Complete'}
                            }
                        }
                    },
                    {
                        'name': 'type'
                    }
                ]
            }
        }
        
        config_file = temp_dir / "valid_conditional.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        # Should parse successfully
        result = parser.parse_config(str(config_file))
        assert 'reconciliation' in result
    
    def test_conditional_mapping_invalid_condition_field(self, temp_dir):
        """Test validation when conditional mapping references non-existent field."""
        parser = ConfigParser()
        
        # Invalid config - condition_field doesn't exist in fields or keys
        invalid_config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'nonexistent_field',
                            'mappings': {
                                'VALUE1': {'old': 'new'}
                            }
                        }
                    }
                ]
            }
        }
        
        config_file = temp_dir / "invalid_conditional.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="references non-existent condition field"):
            parser.parse_config(str(config_file))
    
    def test_conditional_mapping_with_regular_mapping(self, temp_dir):
        """Test validation when field has both regular and conditional mapping."""
        parser = ConfigParser()
        
        # Invalid config - field has both mapping and conditional_mapping
        invalid_config = {
            'reconciliation': {
                'keys': ['id'],
                'fields': [
                    {
                        'name': 'status',
                        'mapping': {'N': 'New'},
                        'conditional_mapping': {
                            'condition_field': 'type',
                            'mappings': {
                                'EQUITY': {'N': 'New'}
                            }
                        }
                    },
                    {
                        'name': 'type'
                    }
                ]
            }
        }
        
        config_file = temp_dir / "both_mappings.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="cannot have both 'mapping' and 'conditional_mapping'"):
            parser.parse_config(str(config_file))
    
    def test_conditional_mapping_with_key_as_condition(self, temp_dir):
        """Test conditional mapping where condition field is a reconciliation key."""
        parser = ConfigParser()
        
        # Valid config - condition_field is one of the keys
        valid_config = {
            'reconciliation': {
                'keys': ['id', 'type'],
                'fields': [
                    {
                        'name': 'status',
                        'conditional_mapping': {
                            'condition_field': 'type',  # This is a key
                            'mappings': {
                                'EQUITY': {'N': 'New'},
                                'BOND': {'N': 'New Order'}
                            }
                        }
                    }
                ]
            }
        }
        
        config_file = temp_dir / "conditional_with_key.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        # Should parse successfully
        result = parser.parse_config(str(config_file))
        assert 'reconciliation' in result
