from ..image.arniqa import ARNIQA
from .image import VideoImageMeasurement


class VideoARNIQA(VideoImageMeasurement):
    """
    A measurement that uses ARNIQA to measure a video.
    """

    image_cls = ARNIQA
    name = "video_arniqa"
    aliases = ["arnvqa"]
