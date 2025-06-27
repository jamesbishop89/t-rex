"""
Reconciliation Engine Module

Core reconciliation logic that compares source and target datasets based on 
configuration rules. Handles field mappings, transformations, and tolerances.
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np

from src.logger_setup import LoggerMixin


class ReconciliationEngine(LoggerMixin):
    """
    Core engine for data reconciliation between source and target datasets.
    
    This class implements the main reconciliation logic including:
    - Data preprocessing with mappings and transformations
    - Tolerance-based comparison for numeric fields
    - Identification of matched, different, and missing records
    - Statistical analysis of reconciliation results
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reconciliation engine with configuration.
        
        Args:
            config: Parsed configuration dictionary containing reconciliation rules
        """
        self.config = config
        self.recon_config = config['reconciliation']
        self.keys = self.recon_config['keys']
        self.fields = self.recon_config['fields']
          # Create field lookup for efficient access
        self.field_configs = {field['name']: field for field in self.fields}
        
        # Separate ignored and comparison fields
        self.comparison_fields = [field for field in self.fields if not field.get('ignore', False)]
        self.ignored_fields = [field for field in self.fields if field.get('ignore', False)]
        
        # Initialize transformation tracking
        self.transformations_applied = {
            'source': {},  # field_name -> list of transformations
            'target': {}   # field_name -> list of transformations
        }
        
        self.logger.info(f"Reconciliation engine initialized with {len(self.keys)} keys and {len(self.fields)} fields")
        self.logger.info(f"Fields for comparison: {len(self.comparison_fields)}, ignored fields: {len(self.ignored_fields)}")
    
    def reconcile(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete reconciliation between source and target datasets.
        
        Args:
            source_df: Source dataset
            target_df: Target dataset
            
        Returns:
            Dict[str, Any]: Reconciliation results including matched, different,
            and missing records, plus statistics
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        self.logger.info("Starting reconciliation process")
        
        # Validate input data
        self._validate_input_data(source_df, target_df)
        
        # Preprocess data (apply mappings and transformations)
        source_processed = self._preprocess_data(source_df.copy(), 'source')
        target_processed = self._preprocess_data(target_df.copy(), 'target')
        
        # Create merged dataset for comparison
        merged_df = self._merge_datasets(source_processed, target_processed)
        
        # Perform field-by-field comparison
        comparison_results = self._compare_fields(merged_df)
        
        # Categorize records
        categorized_records = self._categorize_records(merged_df, comparison_results)
        
        # Calculate statistics
        statistics = self._calculate_statistics(
            source_df, target_df, categorized_records, comparison_results
        )
        
        self.logger.info("Reconciliation process completed")
        
        return {
            'records': categorized_records,
            'statistics': statistics,
            'field_comparison': comparison_results,
            'config': self.config,
            'transformations': self.transformations_applied
        }
    
    def _validate_input_data(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
        """
        Validate that input datasets contain required columns.
        
        Args:
            source_df: Source dataset
            target_df: Target dataset
            
        Raises:
            ValueError: If required columns are missing
        """
        # Check reconciliation keys exist in both datasets
        missing_keys_source = [key for key in self.keys if key not in source_df.columns]
        missing_keys_target = [key for key in self.keys if key not in target_df.columns]
        
        if missing_keys_source:
            raise ValueError(f"Missing reconciliation keys in source data: {missing_keys_source}")
        
        if missing_keys_target:
            raise ValueError(f"Missing reconciliation keys in target data: {missing_keys_target}")
          # Check configured fields exist in both datasets
        field_names = [field['name'] for field in self.fields]
        comparison_field_names = [field['name'] for field in self.comparison_fields]
        ignored_field_names = [field['name'] for field in self.ignored_fields]
        
        missing_fields_source = [field for field in field_names if field not in source_df.columns]
        missing_fields_target = [field for field in field_names if field not in target_df.columns]
        
        if missing_fields_source:
            self.logger.warning(f"Missing fields in source data: {missing_fields_source}")
        
        if missing_fields_target:
            self.logger.warning(f"Missing fields in target data: {missing_fields_target}")
        
        if ignored_field_names:
            self.logger.info(f"Ignored fields (will not be compared): {ignored_field_names}")
        
        self.logger.info("Input data validation completed")
    
    def _preprocess_data(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Apply mappings and transformations to dataset.
        
        Processing order:
        1. Apply mappings first (if configured)
        2. Apply transformations second (if configured)
        
        Args:
            df: Dataset to preprocess
            dataset_type: 'source' or 'target' for logging
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        self.logger.info(f"Preprocessing {dataset_type} data")
        
        for field_config in self.fields:
            field_name = field_config['name']
            
            if field_name not in df.columns:
                self.logger.warning(f"Field {field_name} not found in {dataset_type} data")
                continue
            
            # Check if we should apply mappings to this dataset
            apply_to = field_config.get('apply_to', 'both')
            should_apply = apply_to == 'both' or apply_to == dataset_type
            
            # Apply mapping first
            if 'mapping' in field_config and should_apply:
                df = self._apply_mapping(df, field_name, field_config['mapping'], dataset_type)
            elif 'conditional_mapping' in field_config:
                # Check apply_to for conditional mapping (can be nested)
                conditional_apply_to = field_config['conditional_mapping'].get('apply_to', apply_to)
                should_apply_conditional = conditional_apply_to == 'both' or conditional_apply_to == dataset_type
                if should_apply_conditional:
                    df = self._apply_conditional_mapping(df, field_name, field_config['conditional_mapping'], dataset_type)
            
            # Apply transformation second
            if 'transformation' in field_config and should_apply:
                df = self._apply_transformation(df, field_name, field_config['transformation'], dataset_type)
        
        self.logger.info(f"{dataset_type} data preprocessing completed")
        return df
    
    def _apply_mapping(self, df: pd.DataFrame, field_name: str, 
                      mapping: Dict[str, Any], dataset_type: str) -> pd.DataFrame:
        """
        Apply value mapping to a field.
        
        Args:
            df: Dataset to modify
            field_name: Name of field to map
            mapping: Mapping dictionary {old_value: new_value}
            dataset_type: 'source' or 'target' for logging
            
        Returns:
            pd.DataFrame: Dataset with mapping applied
        """
        try:
            if field_name not in df.columns:
                self.logger.warning(f"Field {field_name} not found in {dataset_type} data")
                return df
            
            # Track original values before mapping
            original_series = df[field_name].copy()
            
            # Apply mapping
            df[field_name] = df[field_name].map(mapping).fillna(df[field_name])
            
            # Track transformations for Excel comments
            changes_made = []
            for idx, (original, new) in enumerate(zip(original_series, df[field_name])):
                if pd.notna(original) and pd.notna(new) and str(original) != str(new):
                    changes_made.append({
                        'row_index': idx,
                        'original_value': original,
                        'new_value': new,
                        'transformation_type': 'mapping',
                        'mapping_rule': f"{original} -> {new}"
                    })
            
            if changes_made:
                if dataset_type not in self.transformations_applied:
                    self.transformations_applied[dataset_type] = {}
                if field_name not in self.transformations_applied[dataset_type]:
                    self.transformations_applied[dataset_type][field_name] = []
                self.transformations_applied[dataset_type][field_name].extend(changes_made)
            
            mapped_count = len(changes_made)
            original_values = original_series.nunique()
            mapped_values = df[field_name].nunique()
            
            self.logger.debug(f"Applied mapping to {field_name} in {dataset_type}: "
                            f"{mapped_count} values changed, {original_values} -> {mapped_values} unique values")
            
        except Exception as e:
            self.logger.error(f"Failed to apply mapping to {field_name} in {dataset_type}: {e}")
            raise ValueError(f"Mapping failed for field {field_name}: {e}")
        
        return df
    
    def _apply_transformation(self, df: pd.DataFrame, field_name: str, 
                            transformation: str, dataset_type: str) -> pd.DataFrame:
        """
        Apply lambda transformation to a field.
        
        Args:
            df: Dataset to modify
            field_name: Name of field to transform
            transformation: Lambda function as string
            dataset_type: 'source' or 'target' for logging
            
        Returns:
            pd.DataFrame: Dataset with transformation applied
        """
        try:
            if field_name not in df.columns:
                self.logger.warning(f"Field {field_name} not found in {dataset_type} data")
                return df
            
            # Track original values before transformation
            original_series = df[field_name].copy()
            
            # Compile lambda function
            transform_func = eval(transformation)
            
            # Apply transformation, handling NaN values
            mask = df[field_name].notna()
            df.loc[mask, field_name] = df.loc[mask, field_name].apply(transform_func)
            
            # Track transformations for Excel comments
            changes_made = []
            for idx, (original, new) in enumerate(zip(original_series, df[field_name])):
                if pd.notna(original) and pd.notna(new) and str(original) != str(new):
                    changes_made.append({
                        'row_index': idx,
                        'original_value': original,
                        'new_value': new,
                        'transformation_type': 'transformation',
                        'transformation_rule': transformation
                    })
            
            if changes_made:
                if dataset_type not in self.transformations_applied:
                    self.transformations_applied[dataset_type] = {}
                if field_name not in self.transformations_applied[dataset_type]:
                    self.transformations_applied[dataset_type][field_name] = []
                self.transformations_applied[dataset_type][field_name].extend(changes_made)
            
            self.logger.debug(f"Applied transformation to {field_name} in {dataset_type}: "
                            f"{len(changes_made)} values changed using {transformation}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply transformation to {field_name} in {dataset_type}: {e}")
            raise ValueError(f"Transformation failed for field {field_name}: {e}")
        
        return df
    
    def _merge_datasets(self, source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge source and target datasets on reconciliation keys.
        
        Args:
            source_df: Preprocessed source dataset
            target_df: Preprocessed target dataset
            
        Returns:
            pd.DataFrame: Merged dataset with source_ and target_ prefixed columns
        """
        self.logger.info("Merging datasets for comparison")
        
        # Add prefixes to distinguish source and target columns
        source_prefixed = source_df.add_prefix('source_')
        target_prefixed = target_df.add_prefix('target_')
        
        # Rename key columns back to original names for merging
        key_renames_source = {f'source_{key}': key for key in self.keys}
        key_renames_target = {f'target_{key}': key for key in self.keys}
        
        source_prefixed = source_prefixed.rename(columns=key_renames_source)
        target_prefixed = target_prefixed.rename(columns=key_renames_target)
          # Perform outer join to capture all records
        merged_df = pd.merge(
            source_prefixed, target_prefixed,
            on=self.keys,
            how='outer',
            suffixes=('', '_target_dup'),
            indicator='_merge_indicator'
        )
        
        self.logger.info(f"Datasets merged: {len(merged_df)} total records")
        return merged_df
    
    def _compare_fields(self, merged_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare configured fields between source and target using tolerances.
        Only compares fields that are not marked as ignored.
        
        Args:
            merged_df: Merged dataset with source and target columns
            
        Returns:
            Dict[str, Any]: Field comparison results (only for non-ignored fields)
        """
        self.logger.info("Comparing fields between source and target")
        
        comparison_results = {}
        
        for field_config in self.comparison_fields:  # Only compare non-ignored fields
            field_name = field_config['name']
            source_col = f'source_{field_name}'
            target_col = f'target_{field_name}'
            
            # Skip if columns don't exist in merged data
            if source_col not in merged_df.columns or target_col not in merged_df.columns:
                self.logger.warning(f"Skipping comparison for {field_name}: columns not found")
                continue
            
            # Compare field with tolerance if specified
            tolerance_config = field_config.get('tolerance')
            comparison_result = self._compare_field_values(
                merged_df[source_col], merged_df[target_col], tolerance_config, field_name
            )
            
            comparison_results[field_name] = comparison_result
        
        # Log ignored fields
        if self.ignored_fields:
            ignored_names = [field['name'] for field in self.ignored_fields]
            self.logger.info(f"Ignored fields (not compared): {ignored_names}")
        
        return comparison_results
    
    def _compare_field_values(self, source_series: pd.Series, target_series: pd.Series,
                            tolerance_config: Optional[Union[float, str]], 
                            field_name: str) -> Dict[str, Any]:
        """
        Compare two series of values with optional tolerance.
        
        Args:
            source_series: Source field values
            target_series: Target field values
            tolerance_config: Tolerance configuration (float or percentage string)
            field_name: Name of field being compared
            
        Returns:
            Dict[str, Any]: Comparison results including match indicators
        """
        if tolerance_config is None:
            # Exact comparison
            matches = (source_series == target_series) | (source_series.isna() & target_series.isna())
        else:
            # Tolerance-based comparison
            matches = self._compare_with_tolerance(source_series, target_series, tolerance_config)
        
        # Count matches where both values are present
        both_present = source_series.notna() & target_series.notna()
        matches_count = matches[both_present].sum()
        total_comparable = both_present.sum()
        
        return {
            'matches': matches,
            'matches_count': matches_count,
            'total_comparable': total_comparable,
            'match_rate': matches_count / total_comparable if total_comparable > 0 else 0,
            'tolerance_config': tolerance_config
        }
    
    def _compare_with_tolerance(self, source_series: pd.Series, target_series: pd.Series,
                              tolerance_config: Union[float, str]) -> pd.Series:
        """
        Compare two numeric series with tolerance.
        
        Args:
            source_series: Source values
            target_series: Target values
            tolerance_config: Tolerance configuration
            
        Returns:
            pd.Series: Boolean series indicating matches within tolerance
        """
        try:
            # Convert to numeric if possible
            source_numeric = pd.to_numeric(source_series, errors='coerce')
            target_numeric = pd.to_numeric(target_series, errors='coerce')
            
            if isinstance(tolerance_config, str) and tolerance_config.endswith('%'):
                # Percentage tolerance
                percentage = float(tolerance_config[:-1]) / 100.0
                # Handle zero division by using absolute tolerance for zero values
                abs_diff = np.abs(source_numeric - target_numeric)
                abs_source = np.abs(source_numeric)
                
                # For zero source values, use absolute difference comparison
                zero_source_mask = abs_source == 0
                non_zero_source_mask = ~zero_source_mask
                
                matches = pd.Series(False, index=source_series.index, dtype=bool)
                
                # For non-zero source values, use percentage tolerance
                matches[non_zero_source_mask] = (
                    abs_diff[non_zero_source_mask] / abs_source[non_zero_source_mask] <= percentage
                )
                
                # For zero source values, consider match if target is also zero
                matches[zero_source_mask] = target_numeric[zero_source_mask] == 0
                
            else:
                # Absolute tolerance - add small epsilon to handle floating-point precision
                tolerance_value = float(tolerance_config)
                # Add small epsilon (1e-9) to handle floating-point precision issues
                effective_tolerance = tolerance_value + 1e-9
                matches = np.isclose(source_numeric, target_numeric, atol=effective_tolerance, rtol=0)
                matches = pd.Series(matches, index=source_series.index, dtype=bool)
            
            # Handle NaN comparisons
            both_nan = source_numeric.isna() & target_numeric.isna()
            matches = (matches | both_nan).astype(bool)
            
            return matches
            
        except Exception as e:
            self.logger.warning(f"Tolerance comparison failed, falling back to exact match: {e}")
            exact_matches = (source_series == target_series) | (source_series.isna() & target_series.isna())
            return exact_matches.astype(bool)
    
    def _categorize_records(self, merged_df: pd.DataFrame, 
                          comparison_results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Categorize records into matched, different, and missing categories.
        
        Args:
            merged_df: Merged dataset
            comparison_results: Field comparison results
            
        Returns:
            Dict[str, pd.DataFrame]: Categorized records
        """
        self.logger.info("Categorizing records")
        
        # Identify record categories based on merge indicator
        both_present = merged_df['_merge_indicator'] == 'both'
        only_source = merged_df['_merge_indicator'] == 'left_only'
        only_target = merged_df['_merge_indicator'] == 'right_only'
        
        # For records present in both, determine if they match or differ
        if comparison_results:
            # All fields must match for record to be considered matched
            all_matches = pd.Series(True, index=merged_df.index)
            for field_name, result in comparison_results.items():
                all_matches = all_matches & result['matches']
            
            matched_mask = both_present & all_matches
            different_mask = both_present & ~all_matches
        else:
            # If no fields configured for comparison, consider all as matched
            matched_mask = both_present
            different_mask = pd.Series(False, index=merged_df.index)
        
        # Create categorized dataframes
        categorized = {
            'matched': merged_df[matched_mask].copy(),
            'different': merged_df[different_mask].copy(),
            'missing_in_target': merged_df[only_source].copy(),
            'missing_in_source': merged_df[only_target].copy()
        }
        
        # Clean up categorized dataframes
        for category, df in categorized.items():
            if '_merge_indicator' in df.columns:
                df.drop('_merge_indicator', axis=1, inplace=True)
        
        # Log categorization results
        for category, df in categorized.items():
            self.logger.info(f"{category}: {len(df)} records")
        
        return categorized
    
    def _calculate_statistics(self, source_df: pd.DataFrame, target_df: pd.DataFrame,
                            categorized_records: Dict[str, pd.DataFrame],
                            comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive reconciliation statistics.
        
        Args:
            source_df: Original source dataset
            target_df: Original target dataset
            categorized_records: Categorized reconciliation records
            comparison_results: Field comparison results
            
        Returns:
            Dict[str, Any]: Comprehensive statistics
        """
        stats = {
            'total_source': len(source_df),
            'total_target': len(target_df),
            'matched': len(categorized_records['matched']),
            'different': len(categorized_records['different']),
            'missing_in_source': len(categorized_records['missing_in_source']),
            'missing_in_target': len(categorized_records['missing_in_target'])
        }
        
        # Field-level statistics
        field_stats = {}
        for field_name, result in comparison_results.items():
            field_stats[field_name] = {
                'matches_count': result['matches_count'],
                'total_comparable': result['total_comparable'],
                'match_rate': result['match_rate'],
                'differences_count': result['total_comparable'] - result['matches_count']
            }
        
        stats['field_statistics'] = field_stats
        
        # Calculate percentages
        total_comparable = stats['matched'] + stats['different']
        if total_comparable > 0:
            stats['match_rate'] = stats['matched'] / total_comparable
            stats['difference_rate'] = stats['different'] / total_comparable
        else:
            stats['match_rate'] = 0
            stats['difference_rate'] = 0
        
        self.logger.info(f"Statistics calculated: {stats['matched']} matched, "
                        f"{stats['different']} different, "
                        f"{stats['missing_in_source']} missing in source, "
                        f"{stats['missing_in_target']} missing in target")
        
        return stats

    def _apply_conditional_mapping(self, df: pd.DataFrame, field_name: str, 
                                 conditional_mapping: Dict[str, Any], dataset_type: str) -> pd.DataFrame:
        """
        Apply conditional value mapping to a field based on another field's value.
        
        Args:
            df: Dataset to modify
            field_name: Name of field to map
            conditional_mapping: Dict with 'condition_field', 'mappings', and optional 'condition_type'/'condition_value'
            dataset_type: 'source' or 'target' for logging
            
        Returns:
            pd.DataFrame: Dataset with conditional mapping applied
        """
        try:
            condition_field = conditional_mapping['condition_field']
            mappings = conditional_mapping['mappings']
            condition_type = conditional_mapping.get('condition_type', 'equals')
            condition_value = conditional_mapping.get('condition_value', None)
            condition_list = conditional_mapping.get('condition_list', None)
            
            if condition_field not in df.columns:
                self.logger.warning(f"Condition field {condition_field} not found in {dataset_type} data")
                return df
            
            if field_name not in df.columns:
                self.logger.warning(f"Field {field_name} not found in {dataset_type} data")
                return df
            
            original_values = df[field_name].nunique()
            
            # Create a copy of the field to modify and track original values
            original_series = df[field_name].copy()
            modified_series = df[field_name].copy()
            
            # Track transformations for Excel comments
            changes_made = []
            
            # Get condition field values and convert to string for most operations
            condition_values = df[condition_field]
            
            # Handle different condition types
            if condition_type in ['equals', 'not_equals', 'starts_with', 'not_starts_with', 
                                'ends_with', 'not_ends_with', 'contains', 'not_contains',
                                'regex_match', 'regex_not_match'] and condition_value:
                
                # Convert to string for string operations
                condition_str_values = condition_values.astype(str)
                
                if condition_type == 'equals':
                    condition_mask = condition_str_values == condition_value
                elif condition_type == 'not_equals':
                    condition_mask = condition_str_values != condition_value
                elif condition_type == 'starts_with':
                    condition_mask = condition_str_values.str.startswith(condition_value, na=False)
                elif condition_type == 'not_starts_with':
                    condition_mask = ~condition_str_values.str.startswith(condition_value, na=False)
                elif condition_type == 'ends_with':
                    condition_mask = condition_str_values.str.endswith(condition_value, na=False)
                elif condition_type == 'not_ends_with':
                    condition_mask = ~condition_str_values.str.endswith(condition_value, na=False)
                elif condition_type == 'contains':
                    condition_mask = condition_str_values.str.contains(condition_value, na=False)
                elif condition_type == 'not_contains':
                    condition_mask = ~condition_str_values.str.contains(condition_value, na=False)
                elif condition_type == 'regex_match':
                    condition_mask = condition_str_values.str.match(condition_value, na=False)
                elif condition_type == 'regex_not_match':
                    condition_mask = ~condition_str_values.str.match(condition_value, na=False)
                
                # Apply mapping for the condition
                if 'default' in mappings:
                    field_mapping = mappings['default']
                    for old_value, new_value in field_mapping.items():
                        value_mask = df[field_name] == old_value
                        combined_mask = condition_mask & value_mask
                        
                        if combined_mask.any():
                            modified_series.loc[combined_mask] = new_value
                            self.logger.debug(f"Applied conditional mapping: {condition_field} {condition_type} '{condition_value}' "
                                            f"-> {field_name}: {old_value} -> {new_value} "
                                            f"({combined_mask.sum()} rows)")
                            
            elif condition_type in ['less_than', 'less_than_equal', 'greater_than', 'greater_than_equal'] and condition_value:
                try:
                    # Convert condition_value to appropriate numeric type
                    numeric_condition_value = pd.to_numeric(condition_value)
                    numeric_condition_values = pd.to_numeric(condition_values, errors='coerce')
                    
                    if condition_type == 'less_than':
                        condition_mask = numeric_condition_values < numeric_condition_value
                    elif condition_type == 'less_than_equal':
                        condition_mask = numeric_condition_values <= numeric_condition_value
                    elif condition_type == 'greater_than':
                        condition_mask = numeric_condition_values > numeric_condition_value
                    elif condition_type == 'greater_than_equal':
                        condition_mask = numeric_condition_values >= numeric_condition_value
                    
                    # Remove NaN values from condition mask
                    condition_mask = condition_mask.fillna(False)
                    
                    # Apply mapping for the condition
                    if 'default' in mappings:
                        field_mapping = mappings['default']
                        for old_value, new_value in field_mapping.items():
                            value_mask = df[field_name] == old_value
                            combined_mask = condition_mask & value_mask
                            
                            if combined_mask.any():
                                modified_series.loc[combined_mask] = new_value
                                self.logger.debug(f"Applied conditional mapping: {condition_field} {condition_type} {condition_value} "
                                                f"-> {field_name}: {old_value} -> {new_value} "
                                                f"({combined_mask.sum()} rows)")
                                
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Could not perform numeric comparison for {condition_type}: {e}")
                    
            elif condition_type in ['in_list', 'not_in_list'] and condition_list:
                condition_str_values = condition_values.astype(str)
                
                if condition_type == 'in_list':
                    condition_mask = condition_str_values.isin([str(v) for v in condition_list])
                elif condition_type == 'not_in_list':
                    condition_mask = ~condition_str_values.isin([str(v) for v in condition_list])
                
                # Apply mapping for the condition
                if 'default' in mappings:
                    field_mapping = mappings['default']
                    for old_value, new_value in field_mapping.items():
                        value_mask = df[field_name] == old_value
                        combined_mask = condition_mask & value_mask
                        
                        if combined_mask.any():
                            modified_series.loc[combined_mask] = new_value
                            self.logger.debug(f"Applied conditional mapping: {condition_field} {condition_type} {condition_list} "
                                            f"-> {field_name}: {old_value} -> {new_value} "
                                            f"({combined_mask.sum()} rows)")
                            
            elif condition_type in ['is_null', 'is_not_null']:
                if condition_type == 'is_null':
                    condition_mask = condition_values.isna()
                elif condition_type == 'is_not_null':
                    condition_mask = condition_values.notna()
                
                # Apply mapping for the condition
                if 'default' in mappings:
                    field_mapping = mappings['default']
                    for old_value, new_value in field_mapping.items():
                        value_mask = df[field_name] == old_value
                        combined_mask = condition_mask & value_mask
                        
                        if combined_mask.any():
                            modified_series.loc[combined_mask] = new_value
                            self.logger.debug(f"Applied conditional mapping: {condition_field} {condition_type} "
                                            f"-> {field_name}: {old_value} -> {new_value} "
                                            f"({combined_mask.sum()} rows)")
            else:
                # Legacy logic: exact matching for backward compatibility
                for condition_value_key, field_mapping in mappings.items():
                    # Find rows where condition field matches the condition value
                    condition_mask = df[condition_field] == condition_value_key
                    
                    if condition_mask.any():
                        # Apply the specific mapping to matching rows
                        for old_value, new_value in field_mapping.items():
                            value_mask = df[field_name] == old_value
                            combined_mask = condition_mask & value_mask
                            
                            if combined_mask.any():
                                modified_series.loc[combined_mask] = new_value
                                self.logger.debug(f"Applied conditional mapping: {condition_field}={condition_value_key} "
                                                f"-> {field_name}: {old_value} -> {new_value} "
                                                f"({combined_mask.sum()} rows)")
            
            # Update the dataframe
            df[field_name] = modified_series
            
            # Track transformations for Excel comments
            for idx, (original, new) in enumerate(zip(original_series, modified_series)):
                if pd.notna(original) and pd.notna(new) and str(original) != str(new):
                    changes_made.append({
                        'row_index': idx,
                        'original_value': original,
                        'new_value': new,
                        'transformation_type': 'conditional',
                        'condition_desc': f"{condition_field} {condition_type} {condition_value or condition_list}"
                    })
            
            if changes_made:
                if dataset_type not in self.transformations_applied:
                    self.transformations_applied[dataset_type] = {}
                if field_name not in self.transformations_applied[dataset_type]:
                    self.transformations_applied[dataset_type][field_name] = []
                self.transformations_applied[dataset_type][field_name].extend(changes_made)
            
            mapped_values = df[field_name].nunique()
            
            self.logger.debug(f"Applied conditional mapping to {field_name} in {dataset_type} "
                            f"based on {condition_field}: {len(changes_made)} values changed, "
                            f"{original_values} -> {mapped_values} unique values")
            
        except Exception as e:
            self.logger.error(f"Failed to apply conditional mapping to {field_name} in {dataset_type}: {e}")
            
        return df
