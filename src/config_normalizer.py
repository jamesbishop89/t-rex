"""
Helpers for normalizing reconciliation config objects into a canonical shape.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_FIELD_BOOLEAN_DEFAULTS = (
    "diff_amount",
    "diff_percent",
    "force_zero_when_settled",
    "hidden",
    "ignore",
)


def normalize_key_config(key_config: Any) -> Dict[str, Any]:
    """Normalize a reconciliation key config into dict form with explicit defaults."""
    if isinstance(key_config, str):
        return {
            "name": key_config,
            "source": key_config,
            "target": key_config,
            "target_alternatives": [],
        }

    normalized = dict(key_config)
    name = normalized["name"]
    normalized.setdefault("source", name)
    normalized.setdefault("target", name)
    normalized.setdefault("target_alternatives", [])
    return normalized


def normalize_conditional_mapping(
    conditional_mapping: Dict[str, Any],
    apply_to: str,
) -> Dict[str, Any]:
    """Normalize nested conditional mapping defaults."""
    normalized = dict(conditional_mapping)
    normalized.setdefault("apply_to", apply_to)

    if "condition_type" in normalized:
        normalized.setdefault("condition_value", None)
        normalized.setdefault("condition_list", None)

    return normalized


def normalize_field_config(field_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a reconciliation field config into a consistent runtime shape."""
    normalized = dict(field_config)
    name = normalized["name"]
    normalized.setdefault("source", name)
    normalized.setdefault("target", name)
    normalized.setdefault("apply_to", "both")

    for field_name in _FIELD_BOOLEAN_DEFAULTS:
        normalized.setdefault(field_name, False)

    if "conditional_mapping" in normalized:
        normalized["conditional_mapping"] = normalize_conditional_mapping(
            normalized["conditional_mapping"],
            normalized["apply_to"],
        )

    return normalized


def normalize_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied config normalized for runtime consumers."""
    normalized = deepcopy(config)
    recon_config = normalized["reconciliation"]
    recon_config["keys"] = [normalize_key_config(key) for key in recon_config["keys"]]
    recon_config["fields"] = [
        normalize_field_config(field_config) for field_config in recon_config["fields"]
    ]
    normalized.setdefault("output", {})
    return normalized
