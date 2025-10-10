from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from time import perf_counter
from typing import Any

from ..annotations import (
    FlexibleMeasurementInputType,
    InputMediaTypeName,
    MultiMediaTypeName,
    ProcessedMeasurementInputType,
)
from ..util import (
    NamedDiscoveryMixin,
    SingletonMixin,
    arrange_input,
    human_readable_duration,
    is_compatible_multi_media_type,
    logger,
    process_media,
    to_multi_media_type,
)


class Measurement(NamedDiscoveryMixin, SingletonMixin, metaclass=ABCMeta):
    """
    A base class for measurements.
    """

    media_type: MultiMediaTypeName
    aggregate: bool = False
    initialized: bool = False

    @classmethod
    def is_compatible(
        cls,
        media_type: InputMediaTypeName,
        aggregate: bool = False,
    ) -> bool:
        """
        Check if a media type is compatible with the measurement.
        """
        return (
            is_compatible_multi_media_type(
                cls.media_type, to_multi_media_type(media_type)
            )
            and cls.aggregate == aggregate
        )

    @classmethod
    def for_media_type(
        cls,
        media_types: InputMediaTypeName | list[InputMediaTypeName],
        aggregate: bool = False,
        name: str | None = None,
        names: list[str] | None = None,
    ) -> list[type[Measurement]] | type[Measurement]:
        """
        :param media_types: The media types to filter by.
        :return: A list of measurements that are compatible with the given media types.
        """
        if not isinstance(media_types, list):
            media_types = [media_types]

        if name is not None:
            if names is not None:
                names.append(name)
            else:
                names = [name]

        matching_measurements = [
            c
            for c in cls.enumerate()
            if any(c.is_compatible(m, aggregate) for m in media_types)
            and (
                names is None
                or c.name in names
                or c.aliases is not None
                and any(n in names for n in c.aliases)
            )
        ]

        if name is not None:
            num_matches = len(matching_measurements)
            if num_matches == 0:
                raise ValueError(
                    f"No measurement found for name {name} and media types {media_types}"
                )
            elif num_matches > 1:
                raise ValueError(
                    f"Multiple measurements found for name {name} and media types {media_types}. Please either use the measurement's unique name, or narrow your search by providing less media types."
                )
            return matching_measurements[0]

        return matching_measurements

    def __init__(self) -> None:
        """
        Initializes the measurement.

        This method will be called several times, but we only have a single instance, so
        we should check if the measurement has already been initialized.
        """
        super().__init__()
        if not self.initialized:
            self.initialized = True
            setup_start = perf_counter()
            self.setup()
            setup_end = perf_counter()
            logger.info(
                f"Setup {type(self).__name__} in {human_readable_duration(setup_end - setup_start)}"
            )

    def setup(self) -> None:
        """
        Setup the measurement. Only called once per class.
        """
        pass

    def calculate(
        self,
        input: FlexibleMeasurementInputType,
        processed: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        :param input: The input to measure.
        :param kwargs: Additional keyword arguments to pass to the measurement. Each measurement class should define its own set of keyword arguments.
        :return: The result of the measurement.
        """
        if isinstance(input, list) or inspect.isgenerator(input):
            if not self.aggregate:
                raise ValueError(
                    f"Measurement {type(self).__name__} does not support aggregated media."
                )

            if not inspect.isgenerator(input):
                assert (
                    len(input) > 1
                ), "Aggregated media must have at least two elements."
                assert all(
                    isinstance(a, tuple) for a in input
                ), "Aggregated media must be a list of tuples."

            if processed:
                data = input
            else:
                data = [tuple(process_media(i) for i in a) for a in input]

            data = [arrange_input(a, self.media_type) for a in data]
        elif isinstance(input, tuple):
            if self.aggregate:
                raise ValueError(
                    f"Measurement {type(self).__name__} does not support non-aggregated media."
                )

            if processed:
                data = input
            else:
                data = tuple(process_media(i) for i in input)

            data = arrange_input(data, self.media_type)

        else:
            raise ValueError(f"Invalid input type: {type(input)}")

        calculate_start = perf_counter()
        result = self(data, **kwargs)
        calculate_end = perf_counter()
        logger.info(
            f"Calculated {type(self).__name__} in {human_readable_duration(calculate_end - calculate_start)}"
        )
        return result

    @abstractmethod
    def __call__(self, input: ProcessedMeasurementInputType, **kwargs: Any) -> Any:
        pass
