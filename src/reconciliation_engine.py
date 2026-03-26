"""
Reconciliation Engine Module

Core reconciliation logic that compares source and target datasets based on
configuration rules. Handles field mappings, transformations, and tolerances.
"""

from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np

from src.config_normalizer import normalize_runtime_config
from src.logger_setup import LoggerMixin
from src.reconciliation_conditions import (
    LIST_CONDITION_ALIASES,
    LIST_CONDITION_TYPES,
    NULL_CONDITION_TYPES,
    NUMERIC_CONDITION_TYPES,
    STRING_CONDITION_TYPES,
)
from src.safe_lambda import compile_config_lambda

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
        self.config = normalize_runtime_config(config)
        self.recon_config = self.config['reconciliation']

        # Process keys configuration
        self.key_configs = self.recon_config['keys']
        self.keys = [key['name'] for key in self.key_configs]

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

        # Initialize post-merge adjustment tracking
        self.post_merge_adjustments = {}  # field_name -> list of {index, original_value, new_value}

        # Preserve helper columns needed for preprocessing/post-merge logic even
        # when they are not part of the displayed reconciliation field set.
        self.auxiliary_fields = set()
        for field in self.fields:
            conditional_mapping = field.get('conditional_mapping')
            if conditional_mapping:
                self.auxiliary_fields.add(conditional_mapping['condition_field'])
            if field.get('force_zero_when_settled', False):
                self.auxiliary_fields.update({'SETTLEMENT_DATE', 'REP_DATE2'})

        self.logger.info(f"Reconciliation engine initialized with {len(self.keys)} keys and {len(self.fields)} fields")
        self.logger.info(f"Fields for comparison: {len(self.comparison_fields)}, ignored fields: {len(self.ignored_fields)}")

    def _execute_config_lambda(self, lambda_str: str, data: Any, context: str) -> Any:
        """Compile and execute a config lambda with consistent error handling."""
        try:
            lambda_func = compile_config_lambda(lambda_str)
            return lambda_func(data)
        except Exception as exc:
            raise ValueError(f"{context}: {exc}") from exc

    def _evaluate_condition(self, df: pd.DataFrame, field: str, condition: str,
                          value: Any = None, values: List[Any] = None) -> pd.Series:
        """
        Evaluate a condition on a dataframe field.

        Args:
            df: DataFrame to evaluate
            field: Field name to check
            condition: Condition type
            value: Single value for comparison
            values: List of values for list-based comparisons

        Returns:
            pd.Series: Boolean mask indicating rows that satisfy the condition
        """
        if field not in df.columns:
            self.logger.warning(f"Field {field} not found for condition evaluation")
            return pd.Series(False, index=df.index)

        series = df[field]

        # String operations
        if condition in STRING_CONDITION_TYPES:

            # Handle numeric equality/inequality for numeric columns
            if condition in ('equals', 'not_equals') and pd.api.types.is_numeric_dtype(series):
                try:
                    num_value = float(value)
                    if condition == 'equals':
                        return series == num_value
                    elif condition == 'not_equals':
                        return series != num_value
                except (ValueError, TypeError):
                    # Fall back to string comparison if value is not numeric
                    pass

            str_series = series.astype(str)
            str_value = str(value) if value is not None else ''

            if condition == 'equals':
                return str_series == str_value
            elif condition == 'not_equals':
                return str_series != str_value
            elif condition == 'starts_with':
                return str_series.str.startswith(str_value, na=False)
            elif condition == 'not_starts_with':
                return ~str_series.str.startswith(str_value, na=False)
            elif condition == 'ends_with':
                return str_series.str.endswith(str_value, na=False)
            elif condition == 'not_ends_with':
                return ~str_series.str.endswith(str_value, na=False)
            elif condition == 'contains':
                return str_series.str.contains(str_value, na=False)
            elif condition == 'not_contains':
                return ~str_series.str.contains(str_value, na=False)
            elif condition == 'regex_match':
                return str_series.str.match(str_value, na=False)
            elif condition == 'regex_not_match':
                return ~str_series.str.match(str_value, na=False)

        # Numeric operations
        elif condition in NUMERIC_CONDITION_TYPES:
            try:
                num_series = pd.to_numeric(series, errors='coerce')
                num_value = float(value)

                if condition == 'less_than':
                    return num_series < num_value
                elif condition == 'less_than_equal':
                    return num_series <= num_value
                elif condition == 'greater_than':
                    return num_series > num_value
                elif condition == 'greater_than_equal':
                    return num_series >= num_value
            except (ValueError, TypeError):
                self.logger.warning(f"Numeric comparison failed for field {field}")
                return pd.Series(False, index=df.index)

        # List operations
        elif condition in (*LIST_CONDITION_TYPES, *LIST_CONDITION_ALIASES) and values:
            str_series = series.astype(str)
            str_values = [str(v) for v in values]

            if condition in ('in_list', 'in'):
                return str_series.isin(str_values)
            elif condition in ('not_in_list', 'not_in'):
                return ~str_series.isin(str_values)

        # Null checks
        elif condition == NULL_CONDITION_TYPES[0]:
            return series.isna()
        elif condition == NULL_CONDITION_TYPES[1]:
            return series.notna()

        return pd.Series(False, index=df.index)

    def _apply_filters(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Apply configured filters to the dataset.

        Args:
            df: Dataset to filter
            dataset_type: 'source' or 'target'

        Returns:
            pd.DataFrame: Filtered dataset
        """
        filters = self.recon_config.get('filters', {}).get(dataset_type, [])

        if not filters:
            return df

        self.logger.info(f"Applying {len(filters)} filters to {dataset_type} data")
        initial_count = len(df)

        for filter_config in filters:
            field = filter_config['field']
            condition = filter_config['condition']
            value = filter_config.get('value')
            values = filter_config.get('values')

            mask = self._evaluate_condition(df, field, condition, value, values)
            df = df[mask]

        final_count = len(df)
        filtered_count = initial_count - final_count

        if filtered_count > 0:
            self.logger.info(f"Filtered {filtered_count} records from {dataset_type} data. "
                           f"Remaining: {final_count}")

        return df

    def _apply_source_key_calculations(self, source_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply source key calculations to compute the source key from multiple columns.

        When a key has 'source_key_calculation', evaluate the lambda against the source
        dataframe to replace the source key column values. This is useful for swaps
        where the source deal_tracking_num needs to be mapped to match the target key.

        Args:
            source_df: Source dataset after filtering

        Returns:
            pd.DataFrame: Source dataset with computed key values
        """
        for key_config in self.key_configs:
            calc = key_config.get('source_key_calculation')
            if not calc:
                continue

            source_col = key_config.get('source', key_config['name'])
            source_df = source_df.copy()
            source_df[source_col] = self._execute_config_lambda(
                calc,
                source_df,
                f"Source key calculation failed for '{source_col}'",
            )
            self.logger.info(f"Applied source key calculation for '{source_col}'")

        return source_df

    def _apply_target_key_calculations(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply target key calculations to compute the target key from multiple columns.

        When a key has 'target_key_calculation', evaluate the lambda against the target
        dataframe to replace the target key column values. This is useful for swaps
        where the key depends on leg direction (e.g., Buy uses DEAL_NUM_1, Sell uses DEAL_NUM_2).

        Args:
            target_df: Target dataset after filtering

        Returns:
            pd.DataFrame: Target dataset with computed key values
        """
        for key_config in self.key_configs:
            calc = key_config.get('target_key_calculation')
            if not calc:
                continue

            target_col = key_config.get('target', key_config['name'])
            target_df = target_df.copy()
            target_df[target_col] = self._execute_config_lambda(
                calc,
                target_df,
                f"Target key calculation failed for '{target_col}'",
            )
            self.logger.info(f"Applied target key calculation for '{target_col}'")

        return target_df

    def _expand_alternative_keys(self, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand target rows for alternative key columns.

        When a key has 'target_alternatives', duplicate target rows so that
        records can be matched on any of the alternative columns. This is useful
        for swaps where near leg is in DEAL_NUM_1 and far leg is in DEAL_NUM_2.

        Args:
            target_df: Target dataset after filtering

        Returns:
            pd.DataFrame: Expanded target dataset
        """
        for key_config in self.key_configs:
            alternatives = key_config.get('target_alternatives', [])
            if not alternatives:
                continue

            primary_target_col = key_config.get('target', key_config['name'])

            # Collect all key values from the primary column
            primary_keys = set(target_df[primary_target_col].dropna())

            frames = [target_df]
            for alt_col in alternatives:
                if alt_col in target_df.columns:
                    # Only create alt rows for values NOT already in primary key set
                    alt_df = target_df[~target_df[alt_col].isin(primary_keys)].copy()
                    if len(alt_df) > 0:
                        alt_df[primary_target_col] = alt_df[alt_col]
                        frames.append(alt_df)
                        self.logger.info(f"Expanded target with alternative key '{alt_col}' -> '{primary_target_col}': {len(alt_df)} additional rows")
                    else:
                        self.logger.info(f"No new unique keys from alternative column '{alt_col}'")
                else:
                    self.logger.warning(f"Alternative key column '{alt_col}' not found in target data")

            if len(frames) > 1:
                target_df = pd.concat(frames, ignore_index=True)
                self.logger.info(f"Target expanded to {len(target_df)} records")

        return target_df

    def _normalize_dataset(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Rename columns in dataset to match canonical field names based on configuration.

        Args:
            df: Dataset to normalize
            dataset_type: 'source' or 'target'

        Returns:
            pd.DataFrame: Dataset with canonical column names
        """
        # Create a new dataframe with only the columns we need, renamed correctly
        new_data = {}

        # Track which columns from original df are used
        used_columns = set()

        # Handle keys
        for key_config in self.key_configs:
            canonical_name = key_config['name']
            # Determine actual column name in the dataset
            actual_name = key_config.get(dataset_type, canonical_name)

            if actual_name in df.columns:
                new_data[canonical_name] = df[actual_name]
                used_columns.add(actual_name)
            else:
                self.logger.warning(f"Key {canonical_name} (mapped to {actual_name}) not found in {dataset_type}")

        # Handle fields
        for field in self.fields:
            canonical_name = field['name']

            # Check for calculations first
            if dataset_type == 'source' and 'source_calculation' in field:
                new_data[canonical_name] = self._execute_config_lambda(
                    field['source_calculation'],
                    df,
                    f"Source field calculation failed for '{canonical_name}'",
                )
                self.logger.debug(f"Calculated field {canonical_name} for source")
                continue

            if dataset_type == 'target' and 'target_calculation' in field:
                new_data[canonical_name] = self._execute_config_lambda(
                    field['target_calculation'],
                    df,
                    f"Target field calculation failed for '{canonical_name}'",
                )
                self.logger.debug(f"Calculated field {canonical_name} for target")
                continue

            # Determine actual column name in the dataset
            actual_name = field.get(dataset_type, canonical_name)

            if actual_name in df.columns:
                new_data[canonical_name] = df[actual_name]
                used_columns.add(actual_name)
            # Note: Missing fields are handled in _preprocess_data (added as NaN)

        # Preserve helper columns used by conditional mappings or post-merge rules.
        for auxiliary_field in self.auxiliary_fields:
            if auxiliary_field in new_data:
                continue
            if auxiliary_field in df.columns:
                new_data[auxiliary_field] = df[auxiliary_field]
                used_columns.add(auxiliary_field)

        # Create new dataframe
        new_df = pd.DataFrame(new_data, index=df.index)

        # Log dropped columns
        dropped_columns = [col for col in df.columns if col not in used_columns]
        if dropped_columns:
            self.logger.info(f"Dropping unconfigured columns from {dataset_type}: {len(dropped_columns)} columns dropped")
            self.logger.debug(f"Dropped columns: {dropped_columns}")

        return new_df

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

        # Strip whitespace from column names to ensure robust matching
        source_df.columns = source_df.columns.str.strip()
        target_df.columns = target_df.columns.str.strip()

        # Apply filters first (on raw data)
        source_df = self._apply_filters(source_df, 'source')
        target_df = self._apply_filters(target_df, 'target')

        # Apply source key calculations (e.g., swap group to FINDUR_ID mapping)
        source_df = self._apply_source_key_calculations(source_df)

        # Apply target key calculations (e.g., near/far leg routing)
        target_df = self._apply_target_key_calculations(target_df)

        # Expand target rows for alternative keys (e.g., near/far leg matching)
        target_df = self._expand_alternative_keys(target_df)

        # Normalize column names based on configuration
        source_normalized = self._normalize_dataset(source_df, 'source')
        target_normalized = self._normalize_dataset(target_df, 'target')

        # Validate input data
        self._validate_input_data(source_normalized, target_normalized)

        # Preprocess data (apply mappings and transformations)
        source_processed = self._preprocess_data(source_normalized.copy(), 'source')
        target_processed = self._preprocess_data(target_normalized.copy(), 'target')

        # Duplicate keys distort reconciliation results and must fail fast.
        self._validate_unique_keys(source_processed, 'source')
        self._validate_unique_keys(target_processed, 'target')

        # Add original index to track back to raw data
        source_processed['__index__'] = source_processed.index
        target_processed['__index__'] = target_processed.index

        # Create merged dataset for comparison
        merged_df = self._merge_datasets(source_processed, target_processed)

        # Apply post-merge calculations (for cross-dataset adjustments)
        merged_df = self._apply_post_merge_calculations(merged_df)

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
            'transformations': self.transformations_applied,
            'post_merge_adjustments': self.post_merge_adjustments,
            'raw_data': {
                'source': source_normalized,
                'target': target_normalized
            }
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
                self.logger.warning(f"Field {field_name} not found in {dataset_type} data. Adding as empty.")
                df[field_name] = np.nan
                # Continue to apply mappings/transformations if they can handle NaN

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
            for row_index, (original, new) in zip(
                original_series.index, zip(original_series, df[field_name])
            ):
                if pd.notna(original) and pd.notna(new) and str(original) != str(new):
                    changes_made.append({
                        'row_index': row_index,
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

    def _apply_masked_mapping(
        self,
        base_series: pd.Series,
        modified_series: pd.Series,
        condition_mask: pd.Series,
        field_mapping: Dict[str, Any],
        field_name: str,
        dataset_type: str,
        condition_desc: str,
        row_condition_descriptions: Optional[Dict[Any, str]] = None,
    ) -> int:
        """Apply a mapping to rows selected by a condition mask."""
        changes = 0

        for old_value, new_value in field_mapping.items():
            value_mask = base_series == old_value
            combined_mask = condition_mask & value_mask

            if not combined_mask.any():
                continue

            modified_series.loc[combined_mask] = new_value
            matched_rows = int(combined_mask.sum())
            changes += matched_rows

            if row_condition_descriptions is not None:
                for row_index in modified_series.index[combined_mask]:
                    row_condition_descriptions[row_index] = condition_desc

            self.logger.debug(
                f"Applied conditional mapping in {dataset_type}: {condition_desc} -> "
                f"{field_name}: {old_value} -> {new_value} ({matched_rows} rows)"
            )

        return changes

    def _validate_unique_keys(self, df: pd.DataFrame, dataset_type: str) -> None:
        """
        Validate that reconciliation keys are unique before merging.

        Duplicate keys create many-to-many joins that inflate reconciliation
        counts and make the result set ambiguous.
        """
        if df.empty:
            return

        key_rows = df[self.keys].copy()
        complete_key_mask = self._get_complete_key_mask(key_rows)
        keyed_rows = key_rows.loc[complete_key_mask]
        duplicate_mask = keyed_rows.duplicated(subset=self.keys, keep=False)
        if not duplicate_mask.any():
            return

        duplicate_rows = keyed_rows.loc[duplicate_mask, self.keys].copy()
        example_keys = duplicate_rows.drop_duplicates().head(5).to_dict('records')
        raise ValueError(
            f"Duplicate reconciliation keys found in {dataset_type} data for keys {self.keys}. "
            f"Example duplicate keys: {example_keys}"
        )

    def _get_complete_key_mask(self, df: pd.DataFrame) -> pd.Series:
        """Return rows where every reconciliation key is populated with a non-blank value."""
        complete_key_mask = pd.Series(True, index=df.index, dtype=bool)
        for key in self.keys:
            key_values = df[key]
            complete_key_mask &= key_values.notna() & key_values.astype(str).str.strip().ne("")
        return complete_key_mask

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
            transform_func = compile_config_lambda(transformation)

            # Apply transformation, handling NaN values
            mask = df[field_name].notna()
            df.loc[mask, field_name] = df.loc[mask, field_name].apply(transform_func)

            # Track transformations for Excel comments
            changes_made = []
            for row_index, (original, new) in zip(
                original_series.index, zip(original_series, df[field_name])
            ):
                if pd.notna(original) and pd.notna(new) and str(original) != str(new):
                    changes_made.append({
                        'row_index': row_index,
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

        source_complete_mask = self._get_complete_key_mask(source_df)
        target_complete_mask = self._get_complete_key_mask(target_df)

        incomplete_source_rows = int((~source_complete_mask).sum())
        incomplete_target_rows = int((~target_complete_mask).sum())
        if incomplete_source_rows:
            self.logger.info(
                f"Excluding {incomplete_source_rows} source rows with incomplete reconciliation keys from matched merge"
            )
        if incomplete_target_rows:
            self.logger.info(
                f"Excluding {incomplete_target_rows} target rows with incomplete reconciliation keys from matched merge"
            )

        # Add prefixes to distinguish source and target columns
        source_prefixed = source_df.add_prefix('source_')
        target_prefixed = target_df.add_prefix('target_')

        # Rename key columns back to original names for merging
        key_renames_source = {f'source_{key}': key for key in self.keys}
        key_renames_target = {f'target_{key}': key for key in self.keys}

        source_prefixed = source_prefixed.rename(columns=key_renames_source)
        target_prefixed = target_prefixed.rename(columns=key_renames_target)

        source_mergeable = source_prefixed.loc[source_complete_mask].copy()
        target_mergeable = target_prefixed.loc[target_complete_mask].copy()

        source_only_columns = [col for col in source_prefixed.columns if col not in self.keys]
        target_only_columns = [col for col in target_prefixed.columns if col not in self.keys]
        full_columns = self.keys + source_only_columns + target_only_columns + ['_merge_indicator']

        # Perform outer join only on rows with complete keys.
        merged_df = pd.merge(
            source_mergeable, target_mergeable,
            on=self.keys,
            how='outer',
            suffixes=('', '_target_dup'),
            indicator='_merge_indicator',
            validate='one_to_one',
        )

        if merged_df.empty:
            merged_df = pd.DataFrame(columns=full_columns)
        else:
            merged_df = merged_df.reindex(columns=full_columns)

        def build_unmatched_rows(
            prefixed_df: pd.DataFrame,
            mask: pd.Series,
            missing_columns: List[str],
            indicator: str,
        ) -> pd.DataFrame:
            unmatched = prefixed_df.loc[~mask].copy()
            if unmatched.empty:
                return pd.DataFrame(columns=full_columns)

            for column in missing_columns:
                unmatched[column] = np.nan
            unmatched['_merge_indicator'] = indicator
            return unmatched.reindex(columns=full_columns)

        unmatched_source = build_unmatched_rows(
            source_prefixed,
            source_complete_mask,
            target_only_columns,
            'left_only',
        )
        unmatched_target = build_unmatched_rows(
            target_prefixed,
            target_complete_mask,
            source_only_columns,
            'right_only',
        )

        if not unmatched_source.empty or not unmatched_target.empty:
            merge_frames = [frame for frame in (merged_df, unmatched_source, unmatched_target) if not frame.empty]
            if merge_frames:
                merged_df = pd.concat(
                    merge_frames,
                    ignore_index=True,
                    sort=False,
                )

        self.logger.info(f"Datasets merged: {len(merged_df)} total records")
        return merged_df

    def _apply_post_merge_calculations(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-merge calculations that need access to both source and target data.

        This handles fields with 'post_merge_source_calculation' or 'post_merge_target_calculation'
        which can reference columns from both datasets using source_ and target_ prefixes.

        Args:
            merged_df: Merged dataset with source_ and target_ prefixed columns

        Returns:
            pd.DataFrame: Merged dataset with post-merge calculations applied
        """
        for field in self.fields:
            field_name = field['name']
            source_col = f"source_{field_name}"
            target_col = f"target_{field_name}"

            # Handle force_zero_when_settled flag
            if field.get('force_zero_when_settled', False):
                original_values = merged_df[source_col].copy()
                settlement_col = 'target_SETTLEMENT_DATE'
                rep_date_col = 'target_REP_DATE2'

                if settlement_col not in merged_df.columns or rep_date_col not in merged_df.columns:
                    raise ValueError(
                        f"force_zero_when_settled for '{field_name}' requires columns "
                        f"'{settlement_col}' and '{rep_date_col}'"
                    )

                rep_date_parsed = pd.to_datetime(
                    merged_df[rep_date_col], format='%d/%m/%y', errors='coerce'
                ).dt.strftime('%Y-%m-%d')
                dates_match = merged_df[settlement_col] == rep_date_parsed
                target_mtm_zero = pd.to_numeric(merged_df[target_col], errors='coerce') == 0

                merged_df[source_col] = np.where(
                    dates_match & target_mtm_zero, 0, merged_df[source_col]
                )
                self._track_post_merge_adjustments(
                    field_name,
                    'source',
                    original_values,
                    merged_df[source_col],
                    row_index_values=merged_df.get('source___index__'),
                    reason="Settlement date = reporting date and target MTM is 0",
                )
                self.logger.debug(f"Applied force_zero_when_settled for {field_name}")

            # Check for post-merge source calculation
            if 'post_merge_source_calculation' in field:
                original_values = merged_df[source_col].copy()
                merged_df[source_col] = self._execute_config_lambda(
                    field['post_merge_source_calculation'],
                    merged_df,
                    f"Post-merge source calculation failed for '{field_name}'",
                )
                self._track_post_merge_adjustments(
                    field_name,
                    'source',
                    original_values,
                    merged_df[source_col],
                    row_index_values=merged_df.get('source___index__'),
                    reason=field.get('post_merge_comment', 'Post-merge source calculation applied'),
                )
                self.logger.debug(f"Applied post-merge source calculation for {field_name}")

            # Check for post-merge target calculation
            if 'post_merge_target_calculation' in field:
                original_values = merged_df[target_col].copy()
                merged_df[target_col] = self._execute_config_lambda(
                    field['post_merge_target_calculation'],
                    merged_df,
                    f"Post-merge target calculation failed for '{field_name}'",
                )
                self._track_post_merge_adjustments(
                    field_name,
                    'target',
                    original_values,
                    merged_df[target_col],
                    row_index_values=merged_df.get('target___index__'),
                    reason=field.get('post_merge_comment', 'Post-merge target calculation applied'),
                )
                self.logger.debug(f"Applied post-merge target calculation for {field_name}")

        return merged_df

    def _track_post_merge_adjustments(
        self,
        field_name: str,
        dataset_type: str,
        original_values: pd.Series,
        new_values: pd.Series,
        row_index_values: Optional[pd.Series] = None,
        reason: Optional[str] = None,
    ) -> None:
        """
        Track rows where post-merge calculations changed values.

        Args:
            field_name: Name of the field
            dataset_type: 'source' or 'target'
            original_values: Values before the calculation
            new_values: Values after the calculation
            row_index_values: Source/target row ids aligned to the merged rows
            reason: Human-readable reason for the adjustment
        """
        key = f"{dataset_type}_{field_name}"
        adjustments = []

        if row_index_values is not None:
            row_index_values = row_index_values.reindex(original_values.index)

        for idx in original_values.index:
            orig = original_values[idx]
            new = new_values[idx]
            # Compare as strings to handle mixed types; skip if both NaN
            try:
                orig_num = float(str(orig).replace(',', '').strip()) if pd.notna(orig) else None
                new_num = float(str(new).replace(',', '').strip()) if pd.notna(new) else None
            except (ValueError, TypeError):
                orig_num = orig
                new_num = new

            if orig_num != new_num:
                adjustment = {
                    'merged_index': idx,
                    'original_value': orig,
                    'new_value': new,
                    'reason': reason,
                }
                if row_index_values is not None:
                    row_index = row_index_values.loc[idx]
                    if pd.notna(row_index):
                        adjustment['row_index'] = row_index
                adjustments.append(adjustment)

        if adjustments:
            existing = self.post_merge_adjustments.get(key, [])
            existing.extend(adjustments)
            self.post_merge_adjustments[key] = existing
            self.logger.info(f"Post-merge adjustment: {len(adjustments)} rows modified for {key}")

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
        if '_merge_indicator' in merged_df.columns:
            both_records_mask = merged_df['_merge_indicator'] == 'both'
        else:
            both_records_mask = pd.Series(True, index=merged_df.index, dtype=bool)

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
                merged_df[source_col],
                merged_df[target_col],
                tolerance_config,
                field_name,
                both_records_mask
            )

            comparison_results[field_name] = comparison_result

        # Log ignored fields
        if self.ignored_fields:
            ignored_names = [field['name'] for field in self.ignored_fields]
            self.logger.info(f"Ignored fields (not compared): {ignored_names}")

        return comparison_results

    def _compare_field_values(self, source_series: pd.Series, target_series: pd.Series,
                            tolerance_config: Optional[Union[float, str]],
                            field_name: str,
                            scope_mask: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Compare two series of values with optional tolerance.

        Args:
            source_series: Source field values
            target_series: Target field values
            tolerance_config: Tolerance configuration (float or percentage string)
            field_name: Name of field being compared
            scope_mask: Optional boolean mask limiting rows used for field statistics

        Returns:
            Dict[str, Any]: Comparison results including match indicators
        """
        if tolerance_config is None:
            # Exact comparison
            matches = (source_series == target_series) | (source_series.isna() & target_series.isna())
        else:
            # Tolerance-based comparison
            matches = self._compare_with_tolerance(source_series, target_series, tolerance_config)

        if scope_mask is None:
            scope_mask = pd.Series(True, index=source_series.index, dtype=bool)
        else:
            scope_mask = scope_mask.reindex(source_series.index, fill_value=False).astype(bool)

        # Comparable rows are in-scope rows where at least one side has a value.
        # This counts one-sided blank/value pairs as differences, while excluding
        # both-missing rows from match-rate statistics.
        both_missing = source_series.isna() & target_series.isna()
        comparable = scope_mask & ~both_missing
        matches_count = int(matches[comparable].sum())
        total_comparable = int(comparable.sum())

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
                non_zero_matches = (
                    abs_diff[non_zero_source_mask] / abs_source[non_zero_source_mask] <= percentage
                ).fillna(False)
                matches.loc[non_zero_source_mask] = non_zero_matches.to_numpy(dtype=bool)

                # For zero source values, consider match if target is also zero
                zero_matches = (target_numeric[zero_source_mask] == 0).fillna(False)
                matches.loc[zero_source_mask] = zero_matches.to_numpy(dtype=bool)

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

            if field_name not in df.columns:
                self.logger.warning(f"Field {field_name} not found in {dataset_type} data")
                return df

            original_values = df[field_name].nunique()

            # Create a copy of the field to modify and track original values
            original_series = df[field_name].copy()
            modified_series = df[field_name].copy()

            # Track transformations for Excel comments
            changes_made = []
            row_condition_descriptions: Dict[Any, str] = {}

            if 'condition_type' in conditional_mapping:
                condition_mask = self._evaluate_condition(
                    df,
                    condition_field,
                    condition_type,
                    condition_value,
                    condition_list,
                )
                condition_operand = condition_value if condition_value is not None else condition_list
                condition_desc = f"{condition_field} {condition_type} {condition_operand}"
                field_mapping = mappings.get('default', {})
                self._apply_masked_mapping(
                    original_series,
                    modified_series,
                    condition_mask,
                    field_mapping,
                    field_name,
                    dataset_type,
                    condition_desc,
                    row_condition_descriptions,
                )
            else:
                # Legacy logic: exact matching for backward compatibility
                for condition_value_key, field_mapping in mappings.items():
                    condition_mask = self._evaluate_condition(
                        df,
                        condition_field,
                        'equals',
                        condition_value_key,
                    )
                    condition_desc = f"{condition_field} equals {condition_value_key}"
                    self._apply_masked_mapping(
                        original_series,
                        modified_series,
                        condition_mask,
                        field_mapping,
                        field_name,
                        dataset_type,
                        condition_desc,
                        row_condition_descriptions,
                    )

            # Update the dataframe
            df[field_name] = modified_series

            # Track transformations for Excel comments
            for row_index, (original, new) in zip(
                original_series.index, zip(original_series, modified_series)
            ):
                if pd.notna(original) and pd.notna(new) and str(original) != str(new):
                    changes_made.append({
                        'row_index': row_index,
                        'original_value': original,
                        'new_value': new,
                        'transformation_type': 'conditional',
                        'condition_desc': row_condition_descriptions.get(
                            row_index,
                            f"{condition_field} {condition_type} {condition_value or condition_list}",
                        ),
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
            raise ValueError(f"Conditional mapping failed for field {field_name}: {e}") from e

        return df
