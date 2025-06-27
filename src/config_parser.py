"""
Configuration Parser Module

Handles parsing and validation of YAML configuration files for T-Rex reconciliation.
Supports field mappings, transformations, tolerances, and reconciliation keys.
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from datetime import datetime

import yaml
from schema import Schema, And, Or, Optional as SchemaOptional, SchemaError

from src.logger_setup import LoggerMixin


class ConfigParser(LoggerMixin):
    """
    Parser and validator for T-Rex YAML configuration files.
    
    This class handles loading, parsing, and validating YAML configuration files
    that define reconciliation rules, field mappings, transformations, and tolerances.
    
    The configuration schema supports:
    - Reconciliation keys for matching records
    - Field-specific mappings, transformations, and tolerances
    - Validation of lambda expressions and tolerance formats
    """
    
    def __init__(self):
        """Initialize the configuration parser with validation schema."""
        self._setup_validation_schema()
    
    def _setup_validation_schema(self) -> None:
        """
        Set up the schema for validating YAML configuration.
        
        Defines the expected structure and types for configuration files,
        including validation rules for transformations and tolerances.
        """        # Schema for individual field configuration
        field_schema = Schema({
            'name': And(str, len),  # Field name must be non-empty string
            SchemaOptional('mapping'): dict,  # Optional mapping dictionary
            SchemaOptional('apply_to'): And(str, lambda x: x in ['both', 'source', 'target']),  # Which dataset(s) to apply mapping/transformation to
            SchemaOptional('conditional_mapping'): {  # New: conditional mapping based on another field
                'condition_field': And(str, len),  # Field name to check condition against
                'mappings': dict,  # Dictionary where keys are condition values, values are mapping dicts
                SchemaOptional('condition_type'): And(str, lambda x: x in [
                    'equals', 'not_equals', 'starts_with', 'not_starts_with', 'ends_with', 'not_ends_with',
                    'contains', 'not_contains', 'less_than', 'less_than_equal', 'greater_than', 'greater_than_equal',
                    'in_list', 'not_in_list', 'regex_match', 'regex_not_match', 'is_null', 'is_not_null'
                ]),  # Type of condition check
                SchemaOptional('condition_value'): str,  # Value to check against (for specific condition types)
                SchemaOptional('condition_list'): list,  # List of values for 'in_list' and 'not_in_list' conditions
                SchemaOptional('apply_to'): And(str, lambda x: x in ['both', 'source', 'target'])  # Which dataset(s) to apply conditional mapping to
            },
            SchemaOptional('transformation'): And(str, self._validate_lambda),  # Optional lambda string
            SchemaOptional('tolerance'): Or(
                And(float, lambda x: x >= 0),  # Positive float for absolute tolerance
                And(str, self._validate_percentage_tolerance)  # Percentage string like "1%"
            ),
            SchemaOptional('ignore'): bool  # Optional flag to ignore field in comparison
        })
          # Main configuration schema
        self.config_schema = Schema({
            'reconciliation': {
                'keys': And(list, lambda x: len(x) > 0),  # Must have at least one key
                'fields': And(list, lambda x: len(x) > 0, [field_schema])  # At least one field
            },
            SchemaOptional('output'): {
                SchemaOptional('filename'): And(str, len)  # Optional output filename
            }
        })
    
    def _validate_lambda(self, lambda_str: str) -> bool:
        """
        Validate that a string represents a valid lambda expression.
        
        Args:
            lambda_str: String representation of lambda function
            
        Returns:
            bool: True if valid lambda expression
            
        Raises:
            ValueError: If lambda expression is invalid
        """
        if not lambda_str.strip().startswith('lambda'):
            raise ValueError(f"Transformation must start with 'lambda': {lambda_str}")
        
        try:
            # Try to compile the lambda expression
            compile(lambda_str, '<string>', 'eval')
            return True
        except SyntaxError as e:
            raise ValueError(f"Invalid lambda syntax: {lambda_str} - {e}")
    
    def _validate_percentage_tolerance(self, tolerance_str: str) -> bool:
        """
        Validate percentage tolerance format (e.g., "1%", "0.5%").
        
        Args:
            tolerance_str: String representation of percentage tolerance
            
        Returns:
            bool: True if valid percentage format
            
        Raises:
            ValueError: If percentage format is invalid
        """
        if not tolerance_str.endswith('%'):
            raise ValueError(f"Percentage tolerance must end with '%': {tolerance_str}")
        
        try:
            percentage_value = float(tolerance_str[:-1])
            if percentage_value < 0:
                raise ValueError(f"Percentage tolerance must be non-negative: {tolerance_str}")
            return True
        except ValueError as e:
            if "could not convert" in str(e):
                raise ValueError(f"Invalid percentage format: {tolerance_str}")
            raise
    
    def parse_config(self, config_path: str) -> Dict[str, Any]:
        """
        Parse and validate a YAML configuration file.
        
        Loads the YAML file, validates its structure against the schema,
        and performs additional validation on field configurations.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dict[str, Any]: Parsed and validated configuration
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            SchemaError: If configuration doesn't match expected schema
            ValueError: If configuration contains invalid values
        """
        self.logger.info(f"Parsing configuration file: {config_path}")
        
        # Check if file exists
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            # Load YAML file
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            if config is None:
                raise ValueError("Configuration file is empty")
            
            self.logger.debug(f"Raw configuration loaded: {config}")
            
            # Validate against schema
            validated_config = self.config_schema.validate(config)
            
            # Additional validation
            self._validate_field_names(validated_config)
            self._validate_reconciliation_keys(validated_config)
            self._validate_conditional_mappings(validated_config)
            
            self.logger.info(f"Configuration validated successfully")
            self.logger.info(f"Reconciliation keys: {validated_config['reconciliation']['keys']}")
            self.logger.info(f"Fields configured: {len(validated_config['reconciliation']['fields'])}")
            
            return validated_config
            
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise yaml.YAMLError(f"Invalid YAML format in {config_path}: {e}")
        
        except SchemaError as e:
            self.logger.error(f"Configuration schema validation failed: {e}")
            raise SchemaError(f"Configuration validation failed: {e}")
    
    def _validate_field_names(self, config: Dict[str, Any]) -> None:
        """
        Validate that field names are unique and non-empty.
        
        Args:
            config: Parsed configuration dictionary
            
        Raises:
            ValueError: If field names are not unique or are empty
        """
        fields = config['reconciliation']['fields']
        field_names = [field['name'] for field in fields]
        
        # Check for empty field names
        empty_names = [name for name in field_names if not name.strip()]
        if empty_names:
            raise ValueError("Field names cannot be empty")
        
        # Check for duplicate field names
        if len(field_names) != len(set(field_names)):
            duplicates = [name for name in field_names if field_names.count(name) > 1]
            raise ValueError(f"Duplicate field names found: {set(duplicates)}")
    
    def _validate_reconciliation_keys(self, config: Dict[str, Any]) -> None:
        """
        Validate reconciliation keys are non-empty strings.
        
        Args:
            config: Parsed configuration dictionary
            
        Raises:
            ValueError: If reconciliation keys are invalid
        """
        keys = config['reconciliation']['keys']
        
        # Check that all keys are non-empty strings
        for key in keys:
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"Reconciliation keys must be non-empty strings: {key}")
        
        # Check for duplicate keys
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate reconciliation keys found: {keys}")
    
    def get_field_config(self, config: Dict[str, Any], field_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific field.
        
        Args:
            config: Parsed configuration dictionary
            field_name: Name of the field to get configuration for
            
        Returns:
            Optional[Dict[str, Any]]: Field configuration or None if not found
        """
        fields = config['reconciliation']['fields']
        for field in fields:
            if field['name'] == field_name:
                return field
        return None
    
    def parse_tolerance(self, tolerance: Union[str, float]) -> Dict[str, Any]:
        """
        Parse tolerance configuration into standardized format.
        
        Args:
            tolerance: Tolerance value (float for absolute, string for percentage)
            
        Returns:
            Dict[str, Any]: Parsed tolerance with type and value
            
        Examples:
            0.01 -> {'type': 'absolute', 'value': 0.01}
            "1%" -> {'type': 'percentage', 'value': 0.01}
        """
        if isinstance(tolerance, (int, float)):
            return {
                'type': 'absolute',
                'value': float(tolerance)
            }
        elif isinstance(tolerance, str) and tolerance.endswith('%'):
            percentage_value = float(tolerance[:-1]) / 100.0
            return {
                'type': 'percentage',
                'value': percentage_value
            }
        else:
            raise ValueError(f"Invalid tolerance format: {tolerance}")
    
    def validate_lambda_function(self, lambda_str: str, test_value: Any = "test") -> bool:
        """
        Validate and test a lambda function string.
        
        Args:
            lambda_str: String representation of lambda function
            test_value: Value to test the lambda function with
            
        Returns:
            bool: True if lambda function is valid and executable
            
        Raises:
            ValueError: If lambda function is invalid or fails execution
        """
        try:
            # Compile and execute the lambda function
            lambda_func = eval(lambda_str)
            
            # Test execution with test value
            result = lambda_func(test_value)
            
            self.logger.debug(f"Lambda function validated: {lambda_str} -> {result}")
            return True
            
        except Exception as e:
            raise ValueError(f"Lambda function validation failed: {lambda_str} - {e}")
    
    def get_output_filename_with_timestamp(self, config: Dict[str, Any], default_filename: str = "reconciliation_results") -> str:
        """
        Get output filename from config with timestamp appended.
        
        Args:
            config: Parsed configuration dictionary
            default_filename: Default filename if not specified in config
            
        Returns:
            str: Filename with timestamp in format $filename_YYYYMMDD_HHMMSS.xlsx
        """
        # Get base filename from config or use default
        output_config = config.get('output', {})
        base_filename = output_config.get('filename', default_filename)
        
        # Remove .xlsx extension if present
        if base_filename.endswith('.xlsx'):
            base_filename = base_filename[:-5]
        
        # Generate timestamp in YYYYMMDD_HHMMSS format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine filename with timestamp
        timestamped_filename = f"{base_filename}_{timestamp}.xlsx"
        
        return timestamped_filename
    
    def _validate_conditional_mappings(self, config: Dict[str, Any]) -> None:
        """
        Validate conditional mappings reference existing fields and have valid condition types.
        
        Args:
            config: Parsed configuration dictionary
            
        Raises:
            ValueError: If conditional mapping references non-existent field or has invalid condition type
        """
        fields = config['reconciliation']['fields']
        field_names = [field['name'] for field in fields]
        keys = config['reconciliation']['keys']
        all_available_fields = set(field_names + keys)
        
        for field in fields:
            if 'conditional_mapping' in field:
                condition_mapping = field['conditional_mapping']
                condition_field = condition_mapping['condition_field']
                
                # Check if condition field exists in the configuration
                if condition_field not in all_available_fields:
                    raise ValueError(
                        f"Conditional mapping for field '{field['name']}' references "
                        f"non-existent condition field '{condition_field}'. "
                        f"Available fields: {sorted(all_available_fields)}"
                    )
                
                # Validate that both regular mapping and conditional mapping are not used together
                if 'mapping' in field:
                    raise ValueError(
                        f"Field '{field['name']}' cannot have both 'mapping' and 'conditional_mapping'. "
                        f"Please use only one mapping type."
                    )
                
                # Check if this uses advanced condition format (with condition_type)
                has_condition_type = 'condition_type' in condition_mapping
                
                # Validate advanced condition format
                if has_condition_type:
                    condition_type = condition_mapping['condition_type']
                    condition_value = condition_mapping.get('condition_value')
                    
                    # Validate condition type requirements
                    condition_types_requiring_value = [
                        'equals', 'not_equals', 'starts_with', 'not_starts_with', 'ends_with', 'not_ends_with',
                        'contains', 'not_contains', 'less_than', 'less_than_equal', 'greater_than', 
                        'greater_than_equal', 'regex_match', 'regex_not_match'
                    ]
                    condition_types_requiring_list = ['in_list', 'not_in_list']
                    condition_types_no_value = ['is_null', 'is_not_null']
                    
                    if condition_type in condition_types_requiring_value and not condition_value:
                        raise ValueError(
                            f"Conditional mapping for field '{field['name']}' uses '{condition_type}' "
                            f"condition type but missing required 'condition_value'"
                        )
                    
                    if condition_type in condition_types_requiring_list:
                        condition_list = condition_mapping.get('condition_list')
                        if not condition_list or not isinstance(condition_list, list):
                            raise ValueError(
                                f"Conditional mapping for field '{field['name']}' uses '{condition_type}' "
                                f"condition type but missing required 'condition_list' (must be a list)"
                            )
                    
                    if condition_type in condition_types_no_value and condition_value:
                        raise ValueError(
                            f"Conditional mapping for field '{field['name']}' uses '{condition_type}' "
                            f"condition type but should not have 'condition_value'"
                        )
                    
                    # Validate regex patterns
                    if condition_type in ['regex_match', 'regex_not_match'] and condition_value:
                        try:
                            import re
                            re.compile(condition_value)
                        except re.error as e:
                            raise ValueError(
                                f"Invalid regex pattern in condition_value for field '{field['name']}': {e}"
                            )
                
                    self.logger.debug(f"Validated conditional mapping for field '{field['name']}' "
                                    f"based on condition field '{condition_field}' with type '{condition_type}'")
