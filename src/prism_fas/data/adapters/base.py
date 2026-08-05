from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from prism_fas.config.models import DatasetDefinition
class DatasetAdapter(ABC):
    def __init__(self, root: Path, definition: DatasetDefinition): self.root, self.definition = root, definition
    @abstractmethod
    def records(self): """Return deterministic records or raise a specific mapping error."""
