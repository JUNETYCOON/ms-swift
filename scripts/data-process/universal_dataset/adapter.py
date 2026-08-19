"""Versioned source/target adapter contracts and explicit plugin loading."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Type

from .schema_registry import CURRENT_SCHEMA_VERSION


CURRENT_UDF_VERSION = CURRENT_SCHEMA_VERSION
ADAPTER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class AdapterSpec:
    """Machine-readable compatibility and loss contract for one adapter."""

    name: str
    version: str
    direction: str
    udf_versions: Tuple[str, ...] = (CURRENT_UDF_VERSION,)
    capabilities: Tuple[str, ...] = ()
    loss_contract: Tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not ADAPTER_NAME.fullmatch(self.name):
            raise ValueError("Adapter name must match {}".format(ADAPTER_NAME.pattern))
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Adapter version must be non-empty")
        if self.direction not in ("source", "target"):
            raise ValueError("Adapter direction must be source or target")
        if not self.udf_versions or any(not isinstance(item, str) or not item for item in self.udf_versions):
            raise ValueError("Adapter udf_versions must contain non-empty versions")
        for field_name in ("capabilities", "loss_contract"):
            values = getattr(self, field_name)
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError("Adapter {} entries must be non-empty strings".format(field_name))

    def supports(self, udf_version: str) -> bool:
        return udf_version in self.udf_versions

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "direction": self.direction,
            "udf_versions": list(self.udf_versions),
            "capabilities": list(self.capabilities),
            "loss_contract": list(self.loss_contract),
            "description": self.description,
        }


@dataclass(frozen=True)
class AdapterDiagnostic:
    """Structured, serializable conversion diagnostic."""

    stage: str
    code: str
    severity: str
    message: str
    source_locator: Optional[str] = None
    record_id: Optional[str] = None
    adapter: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in ("info", "warning", "error"):
            raise ValueError("Diagnostic severity must be info, warning, or error")
        if not self.stage or not self.code or not self.message:
            raise ValueError("Diagnostic stage, code, and message must be non-empty")

    def as_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "stage": self.stage,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        for key in ("source_locator", "record_id", "adapter"):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        if self.details:
            value["details"] = dict(self.details)
        return value


@dataclass(frozen=True)
class AdapterContext:
    dataset_name: str
    source_root: Path
    dataset_version: Optional[str] = None
    preserve_raw: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)
    udf_version: str = CURRENT_UDF_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError("dataset_name must be non-empty")


@dataclass(frozen=True)
class TargetContext:
    output_path: Path
    input_base_dir: Optional[Path] = None
    options: Mapping[str, Any] = field(default_factory=dict)
    udf_version: str = CURRENT_UDF_VERSION


@dataclass
class AdapterResult:
    records: List[Dict[str, Any]]
    diagnostics: List[AdapterDiagnostic] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TargetAdapterResult:
    values: List[Dict[str, Any]]
    diagnostics: List[AdapterDiagnostic] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SourceAdapter(ABC):
    """Pure deterministic mapping from decoded source rows to S1-UDF records."""

    spec: AdapterSpec

    @abstractmethod
    def iter_source(
        self, source: Path, context: AdapterContext
    ) -> Iterator[Tuple[str, Mapping[str, Any]]]:
        """Yield stable source IDs and decoded source records."""

    @abstractmethod
    def convert(
        self, source_id: str, value: Mapping[str, Any], context: AdapterContext
    ) -> AdapterResult:
        """Convert one source record without mutating it."""


class TargetAdapter(ABC):
    """Deterministic record projection with an overridable main-process writer."""

    spec: AdapterSpec

    @abstractmethod
    def convert(self, record: Mapping[str, Any], context: TargetContext) -> TargetAdapterResult:
        """Project one validated S1-UDF record to zero or more JSON-serializable values."""

    def write_output(
        self,
        values: Iterable[Mapping[str, Any]],
        context: TargetContext,
        overwrite: bool = False,
    ) -> int:
        """Write ordered projected values and return the number consumed.

        The default is an atomic JSONL writer. Aggregate, multi-file, Parquet, or
        binary targets may override this method, but must consume the iterable in
        order and provide their own rollback/atomic-commit semantics.
        """

        from .io import atomic_text_writer, json_line

        count = 0
        with atomic_text_writer(context.output_path, overwrite=overwrite) as stream:
            for value in values:
                stream.write(json_line(value))
                count += 1
        return count


class AdapterRegistry:
    def __init__(self) -> None:
        self._sources: Dict[str, Type[SourceAdapter]] = {}
        self._targets: Dict[str, Type[TargetAdapter]] = {}

    @staticmethod
    def _spec(adapter_type: Type[Any], direction: str) -> AdapterSpec:
        spec = getattr(adapter_type, "spec", None)
        if not isinstance(spec, AdapterSpec):
            raise ValueError("Adapter classes need an AdapterSpec named spec")
        if spec.direction != direction:
            raise ValueError(
                "Adapter {!r} declares direction {!r}, expected {!r}".format(
                    spec.name, spec.direction, direction
                )
            )
        if not spec.supports(CURRENT_UDF_VERSION):
            raise ValueError(
                "Adapter {!r} does not support S1-UDF {}".format(spec.name, CURRENT_UDF_VERSION)
            )
        return spec

    def register_source(self, adapter_type: Type[SourceAdapter]) -> Type[SourceAdapter]:
        spec = self._spec(adapter_type, "source")
        if spec.name in self._sources:
            raise ValueError("Source adapter {!r} is already registered".format(spec.name))
        self._sources[spec.name] = adapter_type
        return adapter_type

    def register_target(self, adapter_type: Type[TargetAdapter]) -> Type[TargetAdapter]:
        spec = self._spec(adapter_type, "target")
        if spec.name in self._targets:
            raise ValueError("Target adapter {!r} is already registered".format(spec.name))
        self._targets[spec.name] = adapter_type
        return adapter_type

    def register(self, adapter_type: Type[SourceAdapter]) -> Type[SourceAdapter]:
        """Backward-compatible alias for source adapter registration."""

        return self.register_source(adapter_type)

    def create_source(self, name: str) -> SourceAdapter:
        try:
            return self._sources[name]()
        except KeyError as error:
            raise KeyError(
                "Unknown source adapter {!r}; available: {}".format(
                    name, ", ".join(self.names("source"))
                )
            ) from error

    def create_target(self, name: str) -> TargetAdapter:
        try:
            return self._targets[name]()
        except KeyError as error:
            raise KeyError(
                "Unknown target adapter {!r}; available: {}".format(
                    name, ", ".join(self.names("target"))
                )
            ) from error

    def create(self, name: str) -> SourceAdapter:
        """Backward-compatible alias for source adapter construction."""

        return self.create_source(name)

    def names(self, direction: Optional[str] = None) -> List[str]:
        if direction == "source":
            return sorted(self._sources)
        if direction == "target":
            return sorted(self._targets)
        if direction is not None:
            raise ValueError("direction must be source, target, or None")
        return sorted(set(self._sources) | set(self._targets))

    def specs(self, direction: Optional[str] = None) -> List[AdapterSpec]:
        values: List[AdapterSpec] = []
        if direction in (None, "source"):
            values.extend(self._spec(value, "source") for value in self._sources.values())
        if direction in (None, "target"):
            values.extend(self._spec(value, "target") for value in self._targets.values())
        if direction not in (None, "source", "target"):
            raise ValueError("direction must be source, target, or None")
        return sorted(values, key=lambda item: (item.direction, item.name))

    def spec(self, name: str, direction: str) -> AdapterSpec:
        mapping: Mapping[str, Type[Any]]
        if direction == "source":
            mapping = self._sources
        elif direction == "target":
            mapping = self._targets
        else:
            raise ValueError("direction must be source or target")
        try:
            return self._spec(mapping[name], direction)
        except KeyError as error:
            raise KeyError("Unknown {} adapter {!r}".format(direction, name)) from error


REGISTRY = AdapterRegistry()
_LOADED_PLUGINS: Dict[str, ModuleType] = {}


def register_adapter(adapter_type: Type[SourceAdapter]) -> Type[SourceAdapter]:
    return REGISTRY.register_source(adapter_type)


def register_source_adapter(adapter_type: Type[SourceAdapter]) -> Type[SourceAdapter]:
    return REGISTRY.register_source(adapter_type)


def register_target_adapter(adapter_type: Type[TargetAdapter]) -> Type[TargetAdapter]:
    return REGISTRY.register_target(adapter_type)


def load_adapter_plugin(reference: str) -> ModuleType:
    """Import an explicitly named module or Python file exactly once."""

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Plugin reference must be non-empty")
    raw = reference.strip()
    candidate = Path(raw).expanduser()
    if candidate.suffix.lower() == ".py" or candidate.exists():
        path = candidate.resolve()
        if not path.is_file() or path.suffix.lower() != ".py":
            raise ValueError("Plugin file must be an existing .py file: {}".format(path))
        key = "file:" + str(path)
        if key in _LOADED_PLUGINS:
            return _LOADED_PLUGINS[key]
        module_name = "s1_udf_plugin_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError("Cannot create import spec for plugin {}".format(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        _LOADED_PLUGINS[key] = module
        return module
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", raw):
        raise ValueError("Plugin module must be a dotted Python identifier")
    key = "module:" + raw
    if key not in _LOADED_PLUGINS:
        _LOADED_PLUGINS[key] = importlib.import_module(raw)
    return _LOADED_PLUGINS[key]


def load_adapter_plugins(references: Sequence[str]) -> List[ModuleType]:
    return [load_adapter_plugin(reference) for reference in references]


def _non_empty(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError("{} must not be None".format(field_name))
    text = str(value).strip()
    if not text:
        raise ValueError("{} must be non-empty".format(field_name))
    return text


def stable_record_id(dataset_name: str, source_id: Any) -> str:
    return "{}:{}".format(
        _non_empty(dataset_name, "dataset_name"), _non_empty(source_id, "source_id")
    )


def provisional_media_group_id(media_identities: Iterable[Any], fallback: Any) -> str:
    """Build a row-local hint; ``split`` still consolidates shared media globally."""

    identities = []
    for value in media_identities:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            identities.append(text)
    identities = sorted(set(identities))
    if not identities:
        return "text:" + _non_empty(fallback, "fallback")
    digest = hashlib.sha256("\0".join(identities).encode("utf-8")).hexdigest()
    return "media:" + digest


def convert_iterable(
    adapter: SourceAdapter,
    source_records: Iterable[Tuple[str, Mapping[str, Any]]],
    context: AdapterContext,
) -> Iterator[AdapterResult]:
    """Run an adapter in order; parallel runners preserve this ordering contract."""

    if not adapter.spec.supports(context.udf_version):
        raise ValueError(
            "Adapter {!r} does not support S1-UDF {}".format(adapter.spec.name, context.udf_version)
        )
    for source_id, value in source_records:
        result = adapter.convert(_non_empty(source_id, "source_id"), value, context)
        if not isinstance(result, AdapterResult):
            raise TypeError("Source adapter convert() must return AdapterResult")
        yield result
