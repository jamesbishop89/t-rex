#!/usr/bin/env python3
"""
T-Rex: Data Reconciliation Tool

A production-ready Python tool for comparing source and target datasets using YAML 
configuration files. Produces detailed Excel outputs with comprehensive analysis.

Author: James Bishop
Version: 1.0.0
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

from src.config_parser import ConfigParser
from src.data_loader import DataLoader
from src.reconciliation_engine import ReconciliationEngine
from src.excel_generator import ExcelGenerator
from src.logger_setup import setup_logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for T-Rex reconciliation tool.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing source file,
        target file, config file, and output file paths.
    """
    parser = argparse.ArgumentParser(
        description='T-Rex: Data Reconciliation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,        epilog="""
Examples:
  # Using output filename from config (with automatic timestamp)
  python t-rex.py --source data/source.csv --target data/target.csv --config config.yaml
  
  # Override output filename via command line
  python t-rex.py --source data/source.csv --target data/target.csv --config config.yaml --output results.xlsx
  
  # Short form
  python t-rex.py -s source.xlsx -t target.xlsx -c reconcile.yaml -o output.xlsx
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        required=True,
        type=str,
        help='Path to the source data file (CSV, Excel, etc.)'
    )
    
    parser.add_argument(
        '--target', '-t',
        required=True,
        type=str,
        help='Path to the target data file (CSV, Excel, etc.)'
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        type=str,
        help='Path to the YAML configuration file'    )
    
    parser.add_argument(
        '--output', '-o',
        required=False,
        type=str,
        help='Path for the output Excel file (optional - can be specified in config file)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    return parser.parse_args()


def validate_file_paths(args: argparse.Namespace) -> None:
    """
    Validate that all required input files exist. Output path validation 
    will be handled after config parsing since output can come from config.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        FileNotFoundError: If source, target, or config files don't exist
    """
    # Check if source file exists
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {args.source}")
    
    # Check if target file exists
    target_path = Path(args.target)
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {args.target}")
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")


def validate_and_prepare_output_path(output_path: str) -> None:
    """
    Validate that output directory exists and is writable.
    
    Args:
        output_path: Path for the output file
        
    Raises:
        PermissionError: If output directory is not writable    """
    # Check if output directory exists and is writable
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(f"Cannot create output directory: {e}")
    
    if not output_dir.is_dir():
        raise PermissionError(f"Output directory is not a directory: {output_dir}")
    
    # Test write permissions by attempting to create a temporary file
    try:
        test_file = output_dir / "temp_write_test.tmp"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Output directory is not writable: {e}")


def run_reconciliation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute the complete reconciliation process.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dict[str, Any]: Reconciliation results including statistics and execution metadata
        
    Raises:        Exception: Any exception that occurs during processing is logged and re-raised
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        logger.info("Starting T-Rex reconciliation process")
        logger.info(f"Source file: {args.source}")
        logger.info(f"Target file: {args.target}")
        logger.info(f"Config file: {args.config}")
        
        # Step 1: Parse configuration
        logger.info("Step 1: Parsing configuration file")
        config_parser = ConfigParser()
        config = config_parser.parse_config(args.config)
        logger.info(f"Configuration loaded successfully: {len(config.get('fields', []))} fields configured")
        
        # Determine output filename - use CLI arg if provided, otherwise get from config with timestamp
        if args.output:
            output_file = args.output
            logger.info(f"Using output file from command line: {output_file}")
        else:
            output_file = config_parser.get_output_filename_with_timestamp(config)
            logger.info(f"Generated output filename from config: {output_file}")
        
        # Validate output path
        validate_and_prepare_output_path(output_file)
        logger.info(f"Output file: {output_file}")
        
        # Step 2: Load data files
        logger.info("Step 2: Loading data files")
        data_loader = DataLoader()
        source_df = data_loader.load_data(args.source)
        target_df = data_loader.load_data(args.target)
        
        logger.info(f"Source data loaded: {len(source_df)} rows, {len(source_df.columns)} columns")
        logger.info(f"Target data loaded: {len(target_df)} rows, {len(target_df.columns)} columns")
        
        # Step 3: Execute reconciliation
        logger.info("Step 3: Executing reconciliation")
        reconciliation_engine = ReconciliationEngine(config)
        results = reconciliation_engine.reconcile(source_df, target_df)
        
        # Log reconciliation statistics
        stats = results['statistics']
        logger.info(f"Reconciliation completed:")
        logger.info(f"  - Matched records: {stats['matched']}")
        logger.info(f"  - Different records: {stats['different']}")
        logger.info(f"  - Missing in source: {stats['missing_in_source']}")
        logger.info(f"  - Missing in target: {stats['missing_in_target']}")
        
        # Step 4: Generate Excel output
        logger.info("Step 4: Generating Excel output")
        excel_generator = ExcelGenerator()
          # Add execution metadata to results
        execution_time = time.time() - start_time
        metadata = {
            'source_file': str(Path(args.source).absolute()),
            'target_file': str(Path(args.target).absolute()),
            'config_file': str(Path(args.config).absolute()),
            'output_file': str(Path(output_file).absolute()),
            'execution_time': execution_time,
            'recon_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        excel_generator.generate_excel(results, output_file, metadata)
        logger.info(f"Excel output generated: {output_file}")
        
        # Final summary
        logger.info(f"T-Rex reconciliation completed successfully in {execution_time:.2f} seconds")
        
        return {**results, 'metadata': metadata}
        
    except Exception as e:
        logger.error(f"Reconciliation failed: {str(e)}", exc_info=True)
        raise


def main():
    """
    Main entry point for the T-Rex reconciliation tool.
    
    Handles command line argument parsing, logging setup, file validation,
    and orchestrates the complete reconciliation process.
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logger(args.log_level)
        logger = logging.getLogger(__name__)
        
        # Validate file paths
        validate_file_paths(args)
        
        # Run reconciliation
        results = run_reconciliation(args)
        
        # Print summary to console
        stats = results['statistics']
        metadata = results['metadata']
        
        print("\n" + "="*60)
        print("T-REX RECONCILIATION SUMMARY")
        print("="*60)
        print(f"Execution Time: {metadata['execution_time']:.2f} seconds")
        print(f"Output File: {metadata['output_file']}")
        print(f"\nResults:")
        print(f"  Total Source Records: {stats['total_source']:,}")
        print(f"  Total Target Records: {stats['total_target']:,}")
        print(f"\n  Matched Records: {stats['matched']:,}")
        print(f"  Different Records: {stats['different']:,}")
        print(f"  Missing in Source: {stats['missing_in_source']:,}")
        print(f"  Missing in Target: {stats['missing_in_target']:,}")

        print("="*60)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
