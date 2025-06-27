# T-Rex Technical Documentation

## Architecture Overview

T-Rex is designed with a modular architecture that separates concerns and ensures maintainability:

```
t-rex.py (Main CLI)
    ├── ConfigParser (YAML validation)
    ├── DataLoader (File I/O)
    ├── ReconciliationEngine (Core logic)
    ├── ExcelGenerator (Output formatting)
    └── LoggerSetup (Logging infrastructure)
```

## Module Descriptions

### 1. t-rex.py (Main Entry Point)
- **Purpose**: Command-line interface and orchestration
- **Key Functions**:
  - `parse_arguments()`: CLI argument parsing with validation
  - `validate_file_paths()`: File existence and permission checks
  - `run_reconciliation()`: Main workflow orchestration
  - `main()`: Entry point with error handling

### 2. ConfigParser
- **Purpose**: YAML configuration parsing and validation
- **Key Features**:
  - Schema validation using `schema` library
  - Lambda function validation
  - Tolerance format validation (absolute/percentage)
  - Field uniqueness checking
- **Key Methods**:
  - `parse_config()`: Main parsing entry point
  - `_validate_lambda()`: Lambda expression validation
  - `_validate_percentage_tolerance()`: Percentage format validation
  - `parse_tolerance()`: Tolerance standardization

### 3. DataLoader
- **Purpose**: Multi-format data loading with validation
- **Supported Formats**: CSV, Excel (.xlsx/.xls), JSON, Parquet, Pickle
- **Key Features**:
  - Automatic format detection
  - Data preprocessing (null standardization, string trimming)
  - Memory usage monitoring
  - Comprehensive validation
- **Key Methods**:
  - `load_data()`: Universal data loading
  - `preprocess_data()`: Data cleaning
  - `get_data_info()`: Dataset analysis

### 4. ReconciliationEngine
- **Purpose**: Core reconciliation logic
- **Key Features**:
  - Field-by-field comparison
  - Tolerance-based matching (absolute/percentage)
  - Data transformation pipeline (mapping → transformation)
  - Record categorization (matched/different/missing)
- **Key Methods**:
  - `reconcile()`: Main reconciliation workflow
  - `_compare_with_tolerance()`: Tolerance-aware comparison
  - `_apply_mapping()` / `_apply_transformation()`: Data preprocessing
  - `_categorize_records()`: Result classification

### 5. ExcelGenerator
- **Purpose**: Comprehensive Excel report generation
- **Output Sheets**:
  - Summary: Statistics and metadata
  - Matched: Records within tolerance
  - Different: Records with differences (highlighted)
  - Missing in Target: Source-only records
  - Missing in Source: Target-only records
- **Key Features**:
  - Professional formatting
  - Conditional highlighting
  - Frozen headers and filters
  - Chart generation
- **Key Methods**:
  - `generate_excel()`: Main generation workflow
  - `_highlight_differences()`: Conditional formatting
  - `_add_summary_charts()`: Chart creation

### 6. LoggerSetup
- **Purpose**: Centralized logging configuration
- **Features**:
  - Multiple output handlers (console/file)
  - Configurable log levels
  - Structured formatting
  - Logger mixin for classes

## Data Flow

```
Input Files → DataLoader → ReconciliationEngine → ExcelGenerator → Output Excel
     ↓              ↓              ↓                    ↓
Config YAML → ConfigParser → Field Rules → Report Formatting
```

### Detailed Flow:
1. **Configuration Parse**: YAML → Validated Config Object
2. **Data Loading**: Files → Validated DataFrames
3. **Data Preprocessing**: Apply mappings and transformations
4. **Dataset Merging**: Outer join on reconciliation keys
5. **Field Comparison**: Tolerance-aware field-by-field comparison
6. **Record Categorization**: Sort into matched/different/missing buckets
7. **Statistics Calculation**: Generate summary metrics
8. **Excel Generation**: Format and export comprehensive report

## Error Handling Strategy

### Input Validation
- File existence and permissions
- YAML syntax and schema validation
- DataFrame structure validation
- Configuration completeness checks

### Runtime Error Handling
- Graceful handling of missing fields
- Tolerance calculation edge cases (zero division)
- Lambda function execution errors
- Excel generation failures

### Logging Strategy
- INFO: Progress and statistics
- WARNING: Non-fatal issues (missing fields)
- ERROR: Fatal errors with context
- DEBUG: Detailed execution traces

## Performance Considerations

### Memory Optimization
- Vectorized pandas operations
- Minimal data copying
- Efficient DataFrame merging
- Memory usage monitoring

### Processing Optimization
- Single-pass data processing where possible
- Efficient comparison algorithms
- Optimized Excel writing
- Progress logging for large datasets

### Scalability
- Designed for 100K+ row datasets
- Memory-efficient operations
- Configurable batch processing (future enhancement)

## Testing Strategy

### Unit Tests (>80% Coverage)
- All public methods tested
- Edge cases covered (empty data, invalid configs)
- Error conditions validated
- Mock objects for external dependencies

### Integration Tests
- End-to-end workflow validation
- Real file I/O testing
- Multiple configuration scenarios
- Output validation

### Test Organization
```
tests/
├── conftest.py              # Shared fixtures
├── test_config_parser.py    # Configuration tests
├── test_data_loader.py      # Data loading tests
├── test_reconciliation_engine.py  # Core logic tests
├── test_excel_generator.py # Output generation tests
├── test_logger_setup.py    # Logging tests
└── test_integration.py     # End-to-end tests
```

## Configuration Reference

### YAML Schema
```yaml
reconciliation:
  keys: [list]              # Required: Reconciliation key columns
  fields:                   # Required: List of field configurations
    - name: string          # Required: Field name
      mapping: dict         # Optional: Value mapping
      transformation: string # Optional: Lambda function
      tolerance: number|string # Optional: Comparison tolerance
```

### Field Processing Order
1. **Mapping**: Apply value mappings first
2. **Transformation**: Apply lambda functions second
3. **Comparison**: Use tolerance for comparison

### Tolerance Formats
- **Absolute**: `0.01` (numeric value)
- **Percentage**: `"1%"` (string with % suffix)

## Advanced Field Configuration

#### Dataset-Specific Application (`apply_to`)
Fields can be configured to apply mappings and transformations to specific datasets:

```yaml
fields:
  - name: field_name
    mapping: {...}
    apply_to: "source"    # Options: "source", "target", "both" (default)
    
  - name: another_field
    conditional_mapping:
      condition_field: some_field
      apply_to: "target"  # Apply conditional mapping only to target dataset
      mappings: {...}
```

#### Conditional Mapping
Apply different value mappings based on conditions from other fields:

```yaml
fields:
  - name: status
    conditional_mapping:
      condition_field: trade_type          # Field to check condition against
      condition_type: "equals"             # Type of condition (see below)
      condition_value: "EQUITY"            # Value to compare (if required)
      condition_list: ["VAL1", "VAL2"]     # List of values (for in_list/not_in_list)
      apply_to: "both"                     # Dataset application scope
      mappings:
        "default":                         # When condition is true, apply these mappings
          "N": "New"
          "F": "Filled"
```

#### Supported Condition Types

**String Comparison:**
- `equals`: Exact match with condition_value
- `not_equals`: Does not match condition_value
- `starts_with`: Begins with condition_value
- `not_starts_with`: Does not begin with condition_value
- `ends_with`: Ends with condition_value
- `not_ends_with`: Does not end with condition_value
- `contains`: Contains condition_value substring
- `not_contains`: Does not contain condition_value substring

**Numeric Comparison:**
- `less_than`: Numerically less than condition_value
- `less_than_equal`: Numerically less than or equal to condition_value
- `greater_than`: Numerically greater than condition_value
- `greater_than_equal`: Numerically greater than or equal to condition_value

**List Operations:**
- `in_list`: Value is in condition_list
- `not_in_list`: Value is not in condition_list

**Pattern Matching:**
- `regex_match`: Matches regex pattern in condition_value
- `regex_not_match`: Does not match regex pattern in condition_value

**Null Checks:**
- `is_null`: Field value is null/empty/NaN
- `is_not_null`: Field value is not null/empty/NaN

#### Examples

**Business Rule Example:**
```yaml
# Map currency codes differently based on market region
- name: currency_display
  conditional_mapping:
    condition_field: market_region
    condition_type: "equals"
    condition_value: "APAC"
    mappings:
      "default":
        "USD": "US Dollar"
        "JPY": "Japanese Yen"
        "HKD": "Hong Kong Dollar"

# Handle large trades differently
- name: processing_flag
  conditional_mapping:
    condition_field: notional_amount
    condition_type: "greater_than"
    condition_value: "1000000"
    mappings:
      "default":
        "AUTO": "MANUAL_REVIEW"

# Premium client handling
- name: priority_level
  conditional_mapping:
    condition_field: client_tier
    condition_type: "in_list"
    condition_list: ["PREMIUM", "VIP", "INSTITUTIONAL"]
    mappings:
      "default":
        "STANDARD": "HIGH"
        "LOW": "MEDIUM"

# Regex pattern matching for account types
- name: account_category
  conditional_mapping:
    condition_field: account_number
    condition_type: "regex_match"
    condition_value: "^[0-9]{6}[A-Z]{2}$"
    mappings:
      "default":
        "UNKNOWN": "INSTITUTIONAL"
```

## Production Deployment

### Requirements
- Python 3.8+
- See `requirements.txt` for dependencies
- Sufficient memory for dataset size (rule of thumb: 2-3x file size)

### Monitoring
- Comprehensive logging for audit trails
- Performance metrics in logs
- Error tracking and alerting

### Security Considerations
- Input validation prevents code injection
- File path validation prevents directory traversal
- Lambda functions executed in controlled context

## Troubleshooting Guide

### Common Issues
1. **Configuration Errors**: Check YAML syntax and schema
2. **Missing Files**: Verify file paths and permissions
3. **Memory Issues**: Monitor dataset size and available RAM
4. **Performance**: Check for unnecessary data copies

### Debug Mode
```bash
python t-rex.py --log-level DEBUG ...
```

### Validation Steps
1. Test configuration with small dataset
2. Verify field mappings and transformations
3. Check tolerance settings with known differences
4. Validate output format and content
