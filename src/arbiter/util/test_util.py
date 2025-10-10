from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..measurements.base import Measurement

from ..annotations import InputMediaTypeName


def get_test_measurement(
    media_type: InputMediaTypeName,
    unique_name: str,
    alias: str,
    aggregate: bool = False,
) -> Measurement:
    """
    Get a test measurement for the given media type, unique name, and alias.
    Asserts that the measurement is registered by type and alias, and that it is mismatched by type, alias, and unique name.
    """
    from ..measurements.base import Measurement

    cls = Measurement.for_media_type(media_type, name=alias, aggregate=aggregate)
    cls_2 = Measurement.get(unique_name)
    assert cls is not None, f"{unique_name} is not registered by type and alias"
    assert cls_2 is not None, f"{unique_name} is not registered by unique name"
    assert cls is cls_2, f"{unique_name} is mismatched by type, alias, and unique name"
    return cls()


def human_readable_duration(seconds: float) -> str:
    """
    Convert a number of seconds into a human-readable string.
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def human_readable_size(bytes: int, base_10: bool = True) -> str:
    """
    Convert a number of bytes into a human-readable string.
    """
    if base_10:
        base = 1000
    else:
        base = 1024

    if base_10:
        suffixes = ["B", "KB", "MB", "GB"]
    else:
        suffixes = ["B", "KiB", "MiB", "GiB"]

    for suffix in suffixes:
        if bytes < base:
            return f"{bytes:.2f}{suffix}"
        bytes /= base

    return f"{bytes:.2f}{suffixes[-1]}"
