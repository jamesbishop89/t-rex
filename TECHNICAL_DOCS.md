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

## Extension Points

### Adding New File Formats
1. Add loader method to `DataLoader._supported_extensions`
2. Implement `_load_<format>()` method
3. Add format-specific validation

### Custom Transformations
- Lambda functions support arbitrary Python expressions
- Access to full pandas/numpy functionality
- Error handling for invalid transformations

### Output Formats
- Excel generation is modular
- Additional generators can be added (PDF, HTML, etc.)
- Consistent interface for all generators

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
