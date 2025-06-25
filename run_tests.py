#!/usr/bin/env python3
"""
Test runner script for T-Rex reconciliation tool.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    """Run all tests and checks for T-Rex."""
    print("T-Rex Test Runner")
    print("=================")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    success = True
    
    # Run unit tests
    if not run_command("python -m pytest tests/ -v --tb=short", "Unit Tests"):
        success = False
    
    # Run tests with coverage
    if not run_command("python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing", "Coverage Report"):
        success = False
    
    # Check code style (if flake8 is available)
    try:
        if not run_command("python -m flake8 src/ t-rex.py --max-line-length=100", "Code Style Check"):
            print("Note: flake8 not available or found issues")
    except:
        print("Note: flake8 not available")
    
    # Run integration test with example data
    if not run_command("python t-rex.py --source examples/source_data.csv --target examples/target_data.csv --config examples/config.yaml --output test_output.xlsx", "Integration Test"):
        success = False
    
    print(f"\n{'='*60}")
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print('='*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
