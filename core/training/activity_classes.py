"""Activity class catalogue for CSI-based recognition.

The integer label assigned to each class is its index in CLASS_NAMES.
Keep this list stable — trained weights store class names by position.
"""

from __future__ import annotations

from typing import List

CLASS_NAMES: List[str] = [
    "vacío",
    "parado",
    "sentado",
    "caminando",
    "tirado",
]

NUM_CLASSES: int = len(CLASS_NAMES)


def class_index(name: str) -> int:
    return CLASS_NAMES.index(name)


def class_name(idx: int) -> str:
    if 0 <= idx < NUM_CLASSES:
        return CLASS_NAMES[idx]
    return "?"
