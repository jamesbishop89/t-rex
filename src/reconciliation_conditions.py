"""
Shared condition definitions for reconciliation filters and conditional mappings.
"""

from __future__ import annotations


APPLY_TO_VALUES = ("both", "source", "target")

STRING_CONDITION_TYPES = (
    "equals",
    "not_equals",
    "starts_with",
    "not_starts_with",
    "ends_with",
    "not_ends_with",
    "contains",
    "not_contains",
    "regex_match",
    "regex_not_match",
)

NUMERIC_CONDITION_TYPES = (
    "less_than",
    "less_than_equal",
    "greater_than",
    "greater_than_equal",
)

LIST_CONDITION_TYPES = (
    "in_list",
    "not_in_list",
)

LIST_CONDITION_ALIASES = (
    "in",
    "not_in",
)

NULL_CONDITION_TYPES = (
    "is_null",
    "is_not_null",
)

REGEX_CONDITION_TYPES = (
    "regex_match",
    "regex_not_match",
)

CONDITIONAL_MAPPING_CONDITION_TYPES = (
    *STRING_CONDITION_TYPES,
    *NUMERIC_CONDITION_TYPES,
    *LIST_CONDITION_TYPES,
    *NULL_CONDITION_TYPES,
)

FILTER_CONDITION_TYPES = (
    *CONDITIONAL_MAPPING_CONDITION_TYPES,
    *LIST_CONDITION_ALIASES,
)

CONDITION_TYPES_REQUIRING_VALUE = (
    *STRING_CONDITION_TYPES,
    *NUMERIC_CONDITION_TYPES,
)

CONDITION_TYPES_REQUIRING_LIST = (
    *LIST_CONDITION_TYPES,
    *LIST_CONDITION_ALIASES,
)

CONDITION_TYPES_WITHOUT_VALUE = NULL_CONDITION_TYPES

CONDITION_VALUE_TYPES = (str, int, float, bool)
