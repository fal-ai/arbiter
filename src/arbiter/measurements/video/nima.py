from ..image.nima import NIMA
from .image import VideoImageMeasurement


class VideoNIMA(VideoImageMeasurement):
    """
    A measurement that uses NIMA to measure a video.
    """

    image_cls = NIMA
    name = "video_nima"
    aliases = ["video_nima"]
