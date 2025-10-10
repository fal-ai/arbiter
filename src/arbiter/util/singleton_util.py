from __future__ import annotations

import threading
from typing import Any, ClassVar

__all__ = ["SingletonMixin"]


class SingletonMixin:
    """
    A thread-safe mixin that allows classes to be singletons.
    """

    _instance: ClassVar[type[SingletonMixin] | None] = None
    _lock: ClassVar[threading.Lock | None] = None

    @classmethod
    def _get_lock(cls) -> threading.Lock:
        if getattr(cls, "_lock", None) is None:
            cls._lock = threading.Lock()
        return cls._lock

    def __new__(cls, *args: Any, **kwargs: Any) -> type[SingletonMixin]:
        if cls._instance is None:
            with cls._get_lock():
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
