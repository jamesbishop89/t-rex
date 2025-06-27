# T-Rex: Data Reconciliation Tool

T-Rex is a production-ready Python tool for comparing source and target datasets using YAML configuration files. It produces detailed Excel outputs with comprehensive analysis of matches, differences, and missing records.

## Features

- **Flexible Configuration**: YAML-based configuration with support for field mappings, transformations, and tolerances
- **Advanced Conditional Mapping**: Apply different mappings based on complex logical conditions (equals, contains, regex, numeric comparisons, etc.)
- **Dataset-Specific Processing**: Configure mappings and transformations to apply only to source, target, or both datasets
- **Comprehensive Output**: Five Excel sheets with detailed analysis (Summary, Matched, Different, Missing records)
- **Auto-Sized Excel Columns**: All sheets automatically size columns to fit content for optimal readability
- **Performance Optimized**: Designed for large datasets (>1M rows)
- **Production Ready**: Full error handling, logging, and unit test coverage
- **Visual Analytics**: Summary charts and formatted Excel output

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Using output filename from config (with automatic timestamp)
python t-rex.py --source source.csv --target target.csv --config config.yaml

# Override output filename via command line
python t-rex.py --source source.csv --target target.csv --config config.yaml --output output.xlsx
```

### YAML Configuration Example

```yaml
reconciliation:
  keys: [trade_id, account_id]
  fields:
    - name: execution_price
      tolerance: "0.1%"
    - name: trade_status_conditional
      conditional_mapping:
        condition_field: asset_class
        condition_type: "equals"
        condition_value: "EQ"
        mappings:
          "default":
            "N": "New"
            "F": "Filled"
            "PF": "Partially Filled"
output:
  filename: "my_reconciliation_results"  # Base filename (timestamp added automatically)
```

### Advanced Configuration Features

#### Conditional Mapping
Apply different mappings based on field values:
```yaml
- name: settlement_currency
  conditional_mapping:
    condition_field: market_code
    condition_type: "equals"
    condition_value: "US"
    mappings:
      "default":
        "USD": "US Dollar"
```

#### Dataset-Specific Processing  
Apply mappings only to source, target, or both:
```yaml
- name: field_name
  mapping: {...}
  apply_to: "source"  # Options: "source", "target", "both"
```

#### Condition Types
Supports 18 condition types including:
- String operations: `equals`, `starts_with`, `contains`, `regex_match`
- Numeric comparisons: `greater_than`, `less_than_equal`  
- List operations: `in_list`, `not_in_list`
- Null checks: `is_null`, `is_not_null`

*See `examples/config.yaml` and `TECHNICAL_DOCS.md` for comprehensive examples.*

### Configuration Options

#### Output Configuration
Configure the output filename and behavior:
```yaml
output:
  filename: "my_reconciliation_results"  # Base filename (timestamp added automatically)
```

- **filename**: Base name for output file (optional)
- If not specified, defaults to "reconciliation_results"
- Timestamp is automatically appended: `filename_YYYYMMDD_HHMMSS.xlsx`
- CLI `--output` parameter overrides config file setting

## Output Sheets

1. **Summary**: Overview with reconciliation and field statistics
2. **Matched**: Records that match within tolerances
3. **Different**: Records with differences outside tolerances
4. **Missing in Target**: Records in source but not in target
5. **Missing in Source**: Records in target but not in source

## Testing

Run unit tests:
```bash
pytest tests/ -v --cov=src
```

## Performance

- Optimized for datasets >100K rows
- Target execution time <30 seconds for typical datasets
- Memory-efficient pandas operations

## Error Handling

- Validates YAML configuration
- Handles missing or malformed files
- Clear error messages for troubleshooting
- Comprehensive logging

## Requirements

- Python 3.8+
- pandas
- pyyaml
- openpyxl
- matplotlib
- schema
