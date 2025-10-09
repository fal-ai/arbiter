from collections.abc import Generator
from typing import Literal, Union

from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

MediaTypeName = Literal["image", "video", "audio", "text"]
MultiMediaTypeName = tuple[MediaTypeName, ...]
InputMediaTypeName = Union[MediaTypeName, MultiMediaTypeName]

MediaType = Union[Image.Image, list[Image.Image], NDArray, Tensor]
MeasurableType = Union[MediaType, str]
MeasurementInputType = tuple[MeasurableType, ...]
AggregatedMeasurementInputType = Union[
    list[MeasurementInputType], Generator[MeasurementInputType, None, None]
]
FlexibleMeasurementInputType = Union[
    MeasurementInputType, AggregatedMeasurementInputType
]

ProcessedMeasurableType = Union[Tensor, str]
ProcessedMeasurementInputType = tuple[ProcessedMeasurableType, ...]
AggregatedProcessedMeasurementInputType = Union[
    list[ProcessedMeasurementInputType],
    Generator[ProcessedMeasurementInputType, None, None],
]
FlexibleProcessedMeasurementInputType = Union[
    ProcessedMeasurementInputType, AggregatedProcessedMeasurementInputType
]
