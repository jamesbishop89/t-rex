"""
Command-line entry point for T-Rex reconciliation.
"""

import argparse
import logging
import sys
from typing import Any, Dict

from src.logger_setup import setup_logger
from src.reconciliation_runner import ReconciliationRunRequest, ReconciliationRunner


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for T-Rex."""
    parser = argparse.ArgumentParser(
        description="T-Rex: Data Reconciliation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using output filename from config (with automatic timestamp)
  python t-rex.py --source data/source.csv --target data/target.csv --config config.yaml

  # Override output filename via command line
  python t-rex.py --source data/source.csv --target data/target.csv --config config.yaml --output results.xlsx

  # Short form
  python t-rex.py -s source.xlsx -t target.xlsx -c reconcile.yaml -o output.xlsx
        """,
    )

    parser.add_argument(
        "--source",
        "-s",
        required=True,
        type=str,
        help="Path to the source data file (CSV, Excel, etc.)",
    )
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        type=str,
        help="Path to the target data file (CSV, Excel, etc.)",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=False,
        type=str,
        help="Path for the output Excel file (optional - can be specified in config file)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    return parser.parse_args()


def run_reconciliation(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the complete reconciliation process."""
    logger = logging.getLogger(__name__)

    try:
        runner = ReconciliationRunner()
        run_result = runner.run(
            ReconciliationRunRequest(
                source_path=args.source,
                target_path=args.target,
                config_path=args.config,
                output_path=args.output,
            )
        )
        return run_result.to_dict()

    except Exception as exc:
        logger.error(f"Reconciliation failed: {exc}", exc_info=True)
        raise


def main() -> None:
    """Main entry point for the T-Rex reconciliation tool."""
    try:
        args = parse_arguments()
        setup_logger(args.log_level)

        logger = logging.getLogger(__name__)
        results = run_reconciliation(args)

        stats = results["statistics"]
        metadata = results["metadata"]

        print("\n" + "=" * 60)
        print("T-REX RECONCILIATION SUMMARY")
        print("=" * 60)
        print(f"Execution Time: {metadata['execution_time']:.2f} seconds")
        print(f"Output File: {metadata['output_file']}")
        print("\nResults:")
        print(f"  Total Source Records: {stats['total_source']:,}")
        print(f"  Total Target Records: {stats['total_target']:,}")
        print(f"\n  Matched Records: {stats['matched']:,}")
        print(f"  Different Records: {stats['different']:,}")
        print(f"  Missing in Source: {stats['missing_in_source']:,}")
        print(f"  Missing in Target: {stats['missing_in_target']:,}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error: {exc}")
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
