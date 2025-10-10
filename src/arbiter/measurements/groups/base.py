from __future__ import annotations

import inspect
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ...annotations import FlexibleMeasurementInputType
from ...util import NamedDiscoveryMixin, SingletonMixin, flatten, process_media
from ..base import Measurement


class MeasurementGroup(NamedDiscoveryMixin, SingletonMixin):
    """
    A group of measurements.
    """

    measurements: (
        Sequence[type[Measurement] | type[MeasurementGroup]]
        | list[type[Measurement] | type[MeasurementGroup]]
    )
    measurement_instances: Sequence[Measurement | MeasurementGroup] | None = None
    thread_pool: ThreadPoolExecutor | None = None

    @classmethod
    def from_measurements(
        cls,
        measurements: (
            Sequence[type[Measurement] | type[MeasurementGroup]]
            | list[type[Measurement] | type[MeasurementGroup]]
        ),
        name: str | None = None,
    ) -> MeasurementGroup:
        """
        Create a measurement group from a list of measurements.
        """
        measurements = flatten(*measurements)
        if len(measurements) == 0:
            raise ValueError("No measurements provided")

        if name is None:
            name_stub = "DynamicMeasurementGroup"
            name = f"{name_stub}{uuid.uuid4().hex}"

        group_cls = type(name, (cls,), {"measurements": measurements})
        return group_cls()

    def __init__(self) -> None:
        cls = type(self)
        if not getattr(cls, "_initialized", False):
            with cls._get_lock():
                if not getattr(cls, "_initialized", False):
                    cls.measurement_instances = [
                        m() for m in flatten(*cls.measurements)
                    ]
                    cls.thread_pool = ThreadPoolExecutor(
                        max_workers=len(cls.measurement_instances)
                    )
                    cls._initialized = True

    def calculate(
        self,
        input: FlexibleMeasurementInputType,
        processed: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        :param media: The media to measure.
        :param kwargs: Additional keyword arguments to pass to the measurement. Each measurement class should define its own set of keyword arguments.
        :return: The result of the measurement.
        """
        if not processed:
            is_aggregate = isinstance(input, list) or inspect.isgenerator(input)
            if not is_aggregate:
                input = [input]

            input = [tuple(process_media(i) for i in a) for a in input]
            if not is_aggregate:
                input = input[0]

        futures: dict[str, Future] = {}
        results: dict[str, Any] = {}

        for m in self.measurement_instances:

            def _calculate(m: Measurement) -> Any:
                return m.calculate(input, processed=True, **kwargs)

            futures[m.name] = self.thread_pool.submit(_calculate, m)

        for m.name, future in futures.items():
            results[m.name] = future.result()

        return results
