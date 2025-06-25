#!/usr/bin/env python3
"""
Demonstration script for T-Rex reconciliation tool.
Shows complete functionality with sample data.
"""

import pandas as pd
import tempfile
import yaml
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config_parser import ConfigParser
from src.data_loader import DataLoader
from src.reconciliation_engine import ReconciliationEngine
from src.excel_generator import ExcelGenerator
from src.logger_setup import setup_logger


def create_sample_data():
    """Create comprehensive sample datasets for demonstration."""
    
    # Source dataset with various data types and conditions
    source_data = pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
        'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004', 'TXN005', 
                          'TXN006', 'TXN007', 'TXN008', 'TXN009', 'TXN010'],
        'amount': [100.00, 250.50, 75.25, 500.00, 1000.00, 
                  150.75, 300.00, 450.25, 680.50, 825.75],
        'transaction_date': ['2025-01-01 ', '2025-01-02', ' 2025-01-03', '2025-01-04 ', 
                           '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-08', 
                           '2025-01-09', '2025-01-10'],
        'status': ['Completed', 'Pending', 'Completed', 'Failed', 'Completed',
                  'Pending', 'Completed', 'Failed', 'Completed', 'Pending'],
        'currency': ['USD', 'EUR', 'USD', 'GBP', 'USD', 'EUR', 'USD', 'GBP', 'USD', 'EUR'],
        'fees': [2.50, 6.25, 1.88, 12.50, 25.00, 3.77, 7.50, 11.26, 17.01, 20.64]
    })
    
    # Target dataset with some matching records, some differences, and some missing
    target_data = pd.DataFrame({
        'customer_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1011, 1012, 1013],
        'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004', 'TXN005',
                          'TXN006', 'TXN007', 'TXN011', 'TXN012', 'TXN013'],
        'amount': [100.01, 250.49, 75.30, 500.00, 999.95,  # Small differences within tolerance
                  150.75, 350.00, 200.00, 400.00, 600.00],   # Larger difference, new records
        'transaction_date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04',
                           '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-11',
                           '2025-01-12', '2025-01-13'],
        'status': ['C', 'P', 'C', 'F', 'C', 'P', 'C', 'C', 'P', 'F'],  # Mapped values
        'currency': ['USD', 'EUR', 'USD', 'GBP', 'USD', 'EUR', 'USD', 'USD', 'EUR', 'GBP'],
        'fees': [2.51, 6.24, 1.89, 12.50, 25.05, 3.77, 8.00, 5.00, 10.00, 15.00]
    })
    
    return source_data, target_data


def create_comprehensive_config():
    """Create comprehensive configuration showcasing all features."""
    config = {
        'reconciliation': {
            'keys': ['customer_id', 'transaction_id'],
            'fields': [
                {
                    'name': 'amount',
                    'tolerance': 0.10  # 10 cent absolute tolerance
                },
                {
                    'name': 'transaction_date',
                    'transformation': 'lambda x: x.strip()'  # Remove whitespace
                },
                {
                    'name': 'status',
                    'mapping': {
                        'Completed': 'C',
                        'Pending': 'P',
                        'Failed': 'F'
                    }
                },
                {
                    'name': 'currency'  # Exact match required
                },
                {
                    'name': 'fees',
                    'tolerance': '2%'  # 2% percentage tolerance
                }
            ]
        }
    }
    return config


def run_demonstration():
    """Run complete demonstration of T-Rex functionality."""
    print("🦖 T-Rex Reconciliation Tool Demonstration")
    print("=" * 50)
    
    # Setup logging
    setup_logger('INFO')
    
    # Create temporary directory for files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("📊 Creating sample datasets...")
        source_data, target_data = create_sample_data()
        
        # Save datasets
        source_file = temp_path / "demo_source.csv"
        target_file = temp_path / "demo_target.csv"
        source_data.to_csv(source_file, index=False)
        target_data.to_csv(target_file, index=False)
        
        print(f"   Source records: {len(source_data)}")
        print(f"   Target records: {len(target_data)}")
        
        print("\n⚙️  Creating configuration...")
        config = create_comprehensive_config()
        config_file = temp_path / "demo_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"   Keys: {config['reconciliation']['keys']}")
        print(f"   Fields: {len(config['reconciliation']['fields'])}")
        
        print("\n🔍 Running reconciliation process...")
        
        # Step 1: Parse configuration
        print("   1. Parsing configuration...")
        config_parser = ConfigParser()
        parsed_config = config_parser.parse_config(str(config_file))
        
        # Step 2: Load data
        print("   2. Loading datasets...")
        data_loader = DataLoader()
        source_df = data_loader.load_data(str(source_file))
        target_df = data_loader.load_data(str(target_file))
        
        print(f"      Source: {source_df.shape}")
        print(f"      Target: {target_df.shape}")
        
        # Step 3: Run reconciliation
        print("   3. Executing reconciliation...")
        reconciliation_engine = ReconciliationEngine(parsed_config)
        results = reconciliation_engine.reconcile(source_df, target_df)
        
        # Step 4: Generate Excel output
        print("   4. Generating Excel report...")
        output_file = temp_path / "demo_results.xlsx"
        metadata = {
            'source_file': str(source_file.absolute()),
            'target_file': str(target_file.absolute()),
            'config_file': str(config_file.absolute()),
            'execution_time': 1.23,
            'recon_date': '2025-06-24 12:00:00'
        }
        
        excel_generator = ExcelGenerator()
        excel_generator.generate_excel(results, str(output_file), metadata)
        
        # Display results
        print("\n📈 Reconciliation Results:")
        print("=" * 30)
        stats = results['statistics']
        
        print(f"Total Source Records:    {stats['total_source']:,}")
        print(f"Total Target Records:    {stats['total_target']:,}")
        print(f"Matched Records:         {stats['matched']:,}")
        print(f"Different Records:       {stats['different']:,}")
        print(f"Missing in Source:       {stats['missing_in_source']:,}")
        print(f"Missing in Target:       {stats['missing_in_target']:,}")
        
        if stats['matched'] + stats['different'] > 0:
            match_rate = stats['matched'] / (stats['matched'] + stats['different'])
            print(f"Match Rate:              {match_rate:.1%}")
        
        # Field-level statistics
        if 'field_statistics' in stats:
            print("\n📊 Field-Level Statistics:")
            print("-" * 30)
            for field_name, field_stats in stats['field_statistics'].items():
                print(f"{field_name:15s}: {field_stats['matches_count']:3d}/{field_stats['total_comparable']:3d} "
                      f"({field_stats['match_rate']:.1%}) matches")
        
        # Record categories details
        print("\n📋 Record Categories:")
        print("-" * 20)
        records = results['records']
        
        if not records['matched'].empty:
            print(f"✅ Matched Records: {len(records['matched'])}")
            print("   Sample IDs:", list(records['matched']['customer_id'].head(3)))
        
        if not records['different'].empty:
            print(f"⚠️  Different Records: {len(records['different'])}")
            print("   Sample IDs:", list(records['different']['customer_id'].head(3)))
        
        if not records['missing_in_target'].empty:
            print(f"➖ Missing in Target: {len(records['missing_in_target'])}")
            print("   Sample IDs:", list(records['missing_in_target']['customer_id'].head(3)))
        
        if not records['missing_in_source'].empty:
            print(f"➕ Missing in Source: {len(records['missing_in_source'])}")
            print("   Sample IDs:", list(records['missing_in_source']['customer_id'].head(3)))
        
        print(f"\n📁 Excel report generated: {output_file}")
        print("   Sheets created: Summary, Matched, Different, Missing in Target, Missing in Source")
        
        # Copy output file to current directory for inspection
        import shutil
        final_output = Path.cwd() / "t_rex_demo_results.xlsx"
        shutil.copy2(output_file, final_output)
        print(f"   📄 Report copied to: {final_output}")
        
        print("\n🎉 Demonstration completed successfully!")
        print("=" * 50)
        
        return True


def main():
    """Main demonstration entry point."""
    try:
        return run_demonstration()
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
