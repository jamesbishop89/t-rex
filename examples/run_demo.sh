#!/bin/bash
# T-Rex Demo Script

echo "T-Rex Reconciliation Tool Demo"
echo "=============================="

# Basic reconciliation
echo "Running basic reconciliation..."
python t-rex.py \
    --source examples/source_data.csv \
    --target examples/target_data.csv \
    --config examples/config.yaml \
    --output examples/basic_reconciliation_results.xlsx

echo "Basic reconciliation completed. Check examples/basic_reconciliation_results.xlsx"
echo ""

# Advanced reconciliation with different config
echo "For advanced reconciliation, modify the data files to match advanced_config.yaml"
echo "Then run:"
echo "python t-rex.py --source advanced_source.csv --target advanced_target.csv --config examples/advanced_config.yaml --output advanced_results.xlsx"
