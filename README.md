# T-Rex: Data Reconciliation Tool

T-Rex is a production-ready Python tool for comparing source and target datasets using YAML configuration files. It produces detailed Excel outputs with comprehensive analysis of matches, differences, and missing records.

## Features

- **Flexible Configuration**: YAML-based configuration with support for field mappings, transformations, and tolerances
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
  keys: [id]
  fields:
    - name: amount
    - name: status
output:
  filename: "my_reconciliation_results"  # Base filename (timestamp added automatically)
```

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
