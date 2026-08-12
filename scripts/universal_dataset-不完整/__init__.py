"""Stage1 Universal Dataset Format (S1-UDF)."""

from .adapter import AdapterContext, AdapterRegistry, AdapterResult, SourceAdapter
from .validation import ValidationIssue, validate_record

FORMAT_NAME = "s1-udf"
SCHEMA_VERSION = "1.0.0"

__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "ValidationIssue",
    "validate_record",
    "AdapterContext",
    "AdapterRegistry",
    "AdapterResult",
    "SourceAdapter",
]
