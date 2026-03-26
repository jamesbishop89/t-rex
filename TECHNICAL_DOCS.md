# T-Rex Technical Documentation

## Architecture Overview

T-Rex is organized around a shared orchestration service so the packaged CLI,
compatibility wrapper, and tests all execute the same workflow.

```text
src.cli (packaged CLI)
  -> setup_logger()
  -> ReconciliationRunner
     -> ConfigParser
        -> config_normalizer
        -> safe_lambda
     -> DataLoader
     -> ReconciliationEngine
        -> reconciliation_conditions
     -> ExcelGenerator

t-rex.py -> thin compatibility wrapper around src.cli.main()
```

## Module Descriptions

### 1. src.cli
- Purpose: Command-line interface for installed package usage
- Key responsibilities:
  - Parse CLI arguments
  - Initialize logging
  - Invoke `ReconciliationRunner`
  - Print execution summary and exit codes

### 2. ReconciliationRunner
- Purpose: Shared orchestration layer for end-to-end reconciliation runs
- Key responsibilities:
  - Validate input and output paths
  - Parse configuration
  - Load source and target datasets
  - Execute reconciliation
  - Generate Excel output
  - Return structured run metadata and statistics

### 3. ConfigParser, config_normalizer, and safe_lambda
- Purpose: Validate and normalize YAML configuration before runtime use
- Key features:
  - Schema validation using `schema`
  - Reconciliation key and field normalization
  - Tolerance parsing and validation
  - Restricted lambda validation and compilation
- Key methods:
  - `parse_config()`: Main parsing entry point
  - `get_output_filename_with_timestamp()`: Output naming helper
  - `normalize_runtime_config()`: Canonical config shaping

### 4. DataLoader
- Purpose: Multi-format data loading with validation and preprocessing
- Supported formats: CSV, Excel (`.xlsx`/`.xls`), JSON, Parquet, Pickle
- Key features:
  - Automatic format detection
  - Null standardization and string trimming
  - Dataset validation and metadata reporting
- Key methods:
  - `load_data()`: Universal data loading
  - `preprocess_data()`: Data cleaning
  - `get_data_info()`: Dataset analysis

### 5. ReconciliationEngine and reconciliation_conditions
- Purpose: Core comparison logic and condition evaluation
- Key features:
  - Field-by-field comparison
  - Absolute and percentage tolerances
  - Mapping, conditional mapping, and transformation pipelines
  - Record categorization into matched, different, and missing buckets
- Key methods:
  - `reconcile()`: Main reconciliation workflow
  - `_compare_with_tolerance()`: Tolerance-aware comparison
  - `_apply_mapping()`: Static mapping support
  - `_apply_conditional_mapping()`: Condition-based mapping support
  - `_apply_transformation()`: Safe lambda-based transformations

### 6. ExcelGenerator
- Purpose: Excel report generation and output formatting
- Output sheets:
  - Summary
  - Matched
  - Different
  - Missing in Target
  - Missing in Source
- Key features:
  - Structured formatting and highlighting
  - Frozen headers and filters
  - Comment-based transformation traceability
  - Source-column ordering in summary views
- Key methods:
  - `generate_excel()`: Main workbook generation
  - `_highlight_differences()`: Conditional formatting
  - `_add_transformation_comments()`: Transformation traceability

### 7. LoggerSetup
- Purpose: Centralized logging configuration
- Features:
  - Console and file handlers
  - Configurable log levels
  - Structured formatting helpers

## Data Flow

1. `src.cli` parses arguments and initializes logging.
2. `ReconciliationRunner` validates the requested paths.
3. `ConfigParser` loads YAML, validates it, and normalizes runtime config.
4. `DataLoader` loads source and target files into DataFrames.
5. `ReconciliationEngine` applies preprocessing, merges datasets, and compares fields.
6. `ExcelGenerator` writes the workbook and transformation comments.
7. `ReconciliationRunner` returns statistics and execution metadata.

## Error Handling Strategy

### Input Validation
- File existence and permissions
- YAML syntax and schema validation
- DataFrame structure validation
- Configuration completeness checks

### Runtime Error Handling
- Graceful handling of missing fields
- Tolerance edge cases such as zero division
- Controlled evaluation failures for config lambdas
- Excel generation failures with logging context

### Logging Strategy
- INFO: Progress and summary statistics
- WARNING: Recoverable issues such as missing optional fields
- ERROR: Fatal failures with execution context
- DEBUG: Detailed execution traces

## Performance Considerations

### Memory Optimization
- Vectorized pandas operations
- Minimal data copying
- Efficient DataFrame merging
- Deferred Excel formatting until output generation

### Processing Optimization
- Single-pass preprocessing where possible
- Reuse of normalized config structures
- Shared runner used by CLI and tests to reduce drift

### Scalability
- Designed for large datasets
- Memory footprint depends primarily on DataFrame size and merge width
- Logging exposes row and column counts for operational visibility

## Testing Strategy

### Coverage Areas
- Configuration parsing and normalization
- Safe lambda validation
- Data loading and preprocessing
- Reconciliation logic and condition evaluation
- Excel generation and logging
- Runner orchestration and helper utilities
- End-to-end integration flows

### Test Organization
```text
tests/
|-- conftest.py
|-- test_config_normalizer.py
|-- test_config_parser.py
|-- test_config_parser_unit.py
|-- test_data_loader.py
|-- test_excel_generator.py
|-- test_filters.py
|-- test_generate_all_recs.py
|-- test_integration.py
|-- test_logger_setup.py
|-- test_merge_files.py
|-- test_reconciliation_engine.py
|-- test_reconciliation_runner.py
`-- test_safe_lambda.py
```

## Configuration Reference

### YAML Schema
```yaml
reconciliation:
  keys:
    - name: trade_id
      source: trade_id
      target: trade_id
      target_alternatives: []
  fields:
    - name: execution_price
      source: execution_price
      target: execution_price
      mapping: {}
      conditional_mapping: {}
      transformation: "lambda x: x"
      tolerance: "0.1%"
      apply_to: "both"
```

### Field Processing Order
1. Mapping
2. Conditional mapping
3. Transformation
4. Comparison

### Tolerance Formats
- Absolute: `0.01`
- Percentage: `"1%"`

## Advanced Field Configuration

### Dataset-Specific Application (`apply_to`)
Fields can be configured to apply mappings and transformations to specific
datasets:

```yaml
fields:
  - name: field_name
    mapping: {...}
    apply_to: "source"

  - name: another_field
    conditional_mapping:
      condition_field: some_field
      apply_to: "target"
      mappings: {...}
```

### Conditional Mapping
Apply different value mappings based on conditions from other fields:

```yaml
fields:
  - name: status
    conditional_mapping:
      condition_field: trade_type
      condition_type: "equals"
      condition_value: "EQUITY"
      condition_list: ["VAL1", "VAL2"]
      apply_to: "both"
      mappings:
        "default":
          "N": "New"
          "F": "Filled"
```

### Supported Condition Types

String comparison:
- `equals`
- `not_equals`
- `starts_with`
- `not_starts_with`
- `ends_with`
- `not_ends_with`
- `contains`
- `not_contains`

Numeric comparison:
- `less_than`
- `less_than_equal`
- `greater_than`
- `greater_than_equal`

List operations:
- `in_list`
- `not_in_list`

Pattern matching:
- `regex_match`
- `regex_not_match`

Null checks:
- `is_null`
- `is_not_null`

## Production Deployment

### Requirements
- Python 3.8+
- Dependencies from `requirements.txt`
- Sufficient memory for the dataset size

### Monitoring
- Comprehensive logging for audit trails
- Execution time and record counts in summaries
- Error tracking through logger output

### Security Considerations
- Input validation prevents unsafe file usage
- Config lambdas are compiled through a restricted validator
- Output path preparation is centralized in the runner

## Troubleshooting Guide

### Common Issues
1. Configuration errors: check YAML syntax and required fields.
2. Missing files: verify file paths and permissions.
3. Memory pressure: reduce dataset size or increase available RAM.
4. Unexpected comparison results: inspect mappings, transformations, and tolerances.

### Debug Mode
```bash
t-rex --log-level DEBUG ...
```

### Validation Steps
1. Test the configuration with a small dataset.
2. Verify field mappings and transformations.
3. Check tolerance settings with known differences.
4. Validate the output workbook and summary statistics.
