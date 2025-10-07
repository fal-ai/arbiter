from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def is_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def flatten(*enumerables: Iterable[Any]) -> list[Any]:
    """
    Flatten an arbitrary number of enumerables into a single list.
    """
    flattened: list[Any] = []
    for enumerable in enumerables:
        if is_iterable(enumerable):
            flattened.extend(flatten(*enumerable))
        else:
            flattened.append(enumerable)
    return flattened
