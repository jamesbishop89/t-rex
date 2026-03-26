"""
Unit tests for restricted configuration lambda compilation.
"""

import pandas as pd
import pytest

from src.safe_lambda import UnsafeExpressionError, compile_config_lambda


def test_compile_config_lambda_supports_repo_string_transform_patterns():
    """String-oriented config lambdas should still compile and run."""
    func = compile_config_lambda(
        "lambda x: ''.join(filter(str.isalnum, str(x))).upper()"
    )

    assert func(" ab-12 / cd ") == "AB12CD"


def test_compile_config_lambda_supports_dataframe_calculations():
    """DataFrame calculations used by shipped configs should remain supported."""
    func = compile_config_lambda(
        "lambda df: np.where(pd.to_numeric(df['position'], errors='coerce') < 0, 'B', 'S')"
    )
    df = pd.DataFrame({"position": ["-1", "5"]})

    result = func(df)

    assert result.tolist() == ["B", "S"]


def test_compile_config_lambda_supports_timestamp_strftime_runtime():
    """Pandas Timestamp formatting should work inside the restricted runtime."""
    func = compile_config_lambda(
        "lambda x: pd.to_datetime(x, dayfirst=True).strftime('%Y-%m-%d')"
    )

    assert func("17/03/2026") == "2026-03-17"


def test_compile_config_lambda_rejects_unsafe_imports():
    """Unsafe builtin access should be rejected during validation."""
    with pytest.raises(UnsafeExpressionError, match="Unsupported function call: __import__"):
        compile_config_lambda("lambda x: __import__('os').system('whoami')")
