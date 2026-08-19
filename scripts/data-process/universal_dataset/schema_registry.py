"""Central S1-UDF schema-version registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


FORMAT_NAME = "s1-udf"
CURRENT_SCHEMA_VERSION = "1.0.0"
SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")


@dataclass(frozen=True)
class SchemaBundle:
    version: str
    record_schema: Path
    manifest_schema: Path

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("Schema version must be non-empty")
        for path in (self.record_schema, self.manifest_schema):
            if not path.is_file():
                raise ValueError("Schema file does not exist: {}".format(path))


class SchemaRegistry:
    def __init__(self) -> None:
        self._bundles: Dict[str, SchemaBundle] = {}

    def register(self, bundle: SchemaBundle) -> SchemaBundle:
        if bundle.version in self._bundles:
            raise ValueError("Schema version {!r} is already registered".format(bundle.version))
        self._bundles[bundle.version] = bundle
        return bundle

    def get(self, version: str) -> SchemaBundle:
        try:
            return self._bundles[version]
        except KeyError as error:
            raise KeyError(
                "Unsupported S1-UDF schema version {!r}; supported: {}".format(
                    version, ", ".join(self.versions())
                )
            ) from error

    def versions(self) -> List[str]:
        return sorted(self._bundles)


SCHEMA_REGISTRY = SchemaRegistry()
SCHEMA_REGISTRY.register(
    SchemaBundle(
        version=CURRENT_SCHEMA_VERSION,
        record_schema=SCHEMA_DIRECTORY / "record-1.0.0.schema.json",
        manifest_schema=SCHEMA_DIRECTORY / "manifest-1.0.0.schema.json",
    )
)
