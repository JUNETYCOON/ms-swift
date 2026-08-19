"""Stage1 Universal Dataset Format (S1-UDF)."""

from .adapter import (
    AdapterContext,
    AdapterDiagnostic,
    AdapterRegistry,
    AdapterResult,
    AdapterSpec,
    SourceAdapter,
    TargetAdapter,
    TargetAdapterResult,
    TargetContext,
)
from .schema_registry import CURRENT_SCHEMA_VERSION, FORMAT_NAME, SCHEMA_REGISTRY
from .validation import ValidationIssue, validate_record

SCHEMA_VERSION = CURRENT_SCHEMA_VERSION

__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "ValidationIssue",
    "validate_record",
    "AdapterContext",
    "TargetContext",
    "AdapterSpec",
    "AdapterDiagnostic",
    "AdapterRegistry",
    "AdapterResult",
    "TargetAdapterResult",
    "SourceAdapter",
    "TargetAdapter",
    "SCHEMA_REGISTRY",
]
