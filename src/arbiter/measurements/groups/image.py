from ..base import Measurement
from .base import MeasurementGroup


class ImageMeasurementGroup(MeasurementGroup):
    # Measurements for images alone (i.e. reference-free)
    name = "image"
    measurements = [Measurement.for_media_type(("image",))]


class ImageComparisonMeasurementGroup(MeasurementGroup):
    # Measurements for image comparisons
    name = "image_comparison"
    measurements = [
        Measurement.for_media_type(("image", "image")),
    ]


class ImageSetComparisonMeasurementGroup(MeasurementGroup):
    # Measurements for image set comparisons
    name = "image_set_comparison"
    measurements = [
        Measurement.for_media_type(("image", "image"), aggregate=True),
    ]


class LabeledImageMeasurementGroup(MeasurementGroup):
    # Measurements for captioned images
    name = "labeled_image"
    measurements = [
        Measurement.for_media_type(("text", "image")),
    ]
