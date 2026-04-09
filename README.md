# T-Rex: Data Reconciliation Tool

T-Rex is a production-ready Python tool for comparing source and target datasets using YAML configuration files. It produces detailed Excel outputs with comprehensive analysis of matches, differences, and missing records.

## Features

- **Flexible Configuration**: YAML-based configuration with support for field mappings, transformations, and tolerances
- **Advanced Conditional Mapping**: Apply different mappings based on complex logical conditions (equals, contains, regex, numeric comparisons, etc.)
- **Dataset-Specific Processing**: Configure mappings and transformations to apply only to source, target, or both datasets
- **Comprehensive Output**: Five Excel sheets with detailed analysis (Summary, Matched, Different, Missing records)
- **Transformation Tracking**: Excel comments on transformed cells showing before/after values and transformation rules
- **Source File Column Ordering**: Summary sheet displays reconciliation keys and fields in source file column order
- **Auto-Sized Excel Columns**: All sheets automatically size columns to fit content for optimal readability
- **Performance Optimized**: Designed for large datasets (>1M rows)
- **Production Ready**: Full error handling, logging, and unit test coverage
- **Formatted Reporting**: Structured Excel output with highlighting and comments

## Installation

Install runtime dependencies for running from a repository checkout:

```bash
python -m pip install -r requirements.txt
```

Install the package together with development tooling:

```bash
python -m pip install -e ".[dev]"
```

## Usage

### Command Line

```bash
# Installed console script
t-rex --source source.csv --target target.csv --config config.yaml

# Compatibility wrapper when running directly from a repo checkout
python t-rex.py --source source.csv --target target.csv --config config.yaml

# Override output filename via command line
t-rex --source source.csv --target target.csv --config config.yaml --output output.xlsx
```

### Market Data Automation

Use the dedicated market-data automation entrypoint to poll the source and
target folders, wait for files to become stable, pick the newest matching pair,
run the reconciliation, and optionally email the workbook.

```bash
t-rex-market-data \
  --automation-config config/market-data/automation.yaml \
  --source-dir /prod/Murex/outgoing/prod/interfaces/marketdata \
  --target-dir target/dr2/market-data \
  --output-dir output/dr2/market-data \
  --job-groups intraday \
  --once
```

Recommended production pattern on Ubuntu: schedule
`t-rex-market-data --once` every 1-5 minutes with `systemd` or `cron`. That
gives you idempotent retries after restarts and uses the JSON state file in the
output folder to avoid processing the same target extract twice.

The automation config is designed so `--source-dir` points at the common
market-data root, while each job's `source_globs` descend into the relevant
`<job>/ProcessedFiles` or `<job>_eod/ProcessedFiles` directory.

Best production split:
- Run `--job-groups intraday` every few minutes.
- Run `--job-groups eod` once per day after the EOD target extracts are expected.

Example repo checkout command:

```bash
python -m src.market_data_automation \
  --automation-config config/market-data/automation.yaml \
  --source-dir /prod/Murex/outgoing/prod/interfaces/marketdata \
  --target-dir /data/target/market-data \
  --output-dir /data/output/market-data \
  --job-groups intraday \
  --min-file-age-seconds 60 \
  --once
```

Ubuntu deployment templates are included in:
- `examples/market-data-automation.service`
- `examples/market-data-automation.timer`
- `examples/market-data-automation.env`

Optional SMTP delivery can be configured with `--email-to`, `--email-from`,
`--smtp-host`, and related flags, or by setting the matching `TREX_*`
environment variables.

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

If you have only installed runtime dependencies, install the dev extras first:
```bash
python -m pip install -e ".[dev]"
```

Run the test suite:
```bash
python -m pytest tests/ -v --cov=src
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
- schema
