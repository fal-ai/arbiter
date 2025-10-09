from ..base import Measurement
from .base import MeasurementGroup


class VideoMeasurementGroup(MeasurementGroup):
    # Measurements for videos alone (i.e. reference-free)
    name = "video"
    measurements = [Measurement.for_media_type(("video",))]


class VideoComparisonMeasurementGroup(MeasurementGroup):
    # Measurements for video comparisons
    name = "video_comparison"
    measurements = [
        Measurement.for_media_type(("video", "video")),
    ]


class VideoSetComparisonMeasurementGroup(MeasurementGroup):
    # Measurements for video set comparisons
    name = "video_set_comparison"
    measurements = [
        Measurement.for_media_type(("video", "video"), aggregate=True),
    ]


class LabeledVideoMeasurementGroup(MeasurementGroup):
    # Measurements for captioned videos
    name = "labeled_video"
    measurements = [
        Measurement.for_media_type(("text", "video")),
    ]
