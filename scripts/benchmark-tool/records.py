from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EvaluationRecord:
    model: str
    benchmark: str
    metrics: Dict[str, float]
