"""Extension contract for source-specific S1-UDF import adapters."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Type


@dataclass(frozen=True)
class AdapterContext:
    dataset_name: str
    source_root: Path
    dataset_version: Optional[str] = None
    preserve_raw: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AdapterResult:
    records: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)


class SourceAdapter(ABC):
    """One deterministic source adapter, suitable for use in process workers."""

    name: str

    @abstractmethod
    def iter_source(self, source: Path) -> Iterator[Tuple[str, Mapping[str, Any]]]:
        """Yield stable source IDs and decoded source records."""

    @abstractmethod
    def convert(
        self, source_id: str, value: Mapping[str, Any], context: AdapterContext
    ) -> AdapterResult:
        """Convert one source record without mutating it."""


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, Type[SourceAdapter]] = {}

    def register(self, adapter_type: Type[SourceAdapter]) -> Type[SourceAdapter]:
        name = getattr(adapter_type, "name", "")
        if not isinstance(name, str) or not name:
            raise ValueError("Adapter classes need a non-empty name")
        if name in self._adapters:
            raise ValueError("Adapter {!r} is already registered".format(name))
        self._adapters[name] = adapter_type
        return adapter_type

    def create(self, name: str) -> SourceAdapter:
        try:
            return self._adapters[name]()
        except KeyError as error:
            raise KeyError(
                "Unknown adapter {!r}; available: {}".format(name, ", ".join(self.names()))
            ) from error

    def names(self) -> List[str]:
        return sorted(self._adapters)


REGISTRY = AdapterRegistry()


def register_adapter(adapter_type: Type[SourceAdapter]) -> Type[SourceAdapter]:
    return REGISTRY.register(adapter_type)


def stable_record_id(dataset_name: str, source_id: Any) -> str:
    return "{}:{}".format(dataset_name, source_id)


def provisional_media_group_id(media_identities: Iterable[str], fallback: str) -> str:
    """Build a row-local hint; ``split`` still consolidates shared-media components globally."""

    identities = sorted(set(str(value) for value in media_identities if str(value)))
    if not identities:
        return "text:" + fallback
    digest = hashlib.sha256("\0".join(identities).encode("utf-8")).hexdigest()
    return "media:" + digest


def convert_iterable(
    adapter: SourceAdapter,
    source_records: Iterable[Tuple[str, Mapping[str, Any]]],
    context: AdapterContext,
) -> Iterator[AdapterResult]:
    """Run an adapter in order; parallel CLIs must preserve this ordering contract."""

    for source_id, value in source_records:
        yield adapter.convert(source_id, value, context)
