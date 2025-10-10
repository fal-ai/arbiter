from ..image.musiq import MUSIQ
from .image import VideoImageMeasurement


class VideoMUSIQ(VideoImageMeasurement):
    """
    A measurement that uses MUSIQ to measure a video.
    """

    image_cls = MUSIQ
    name = "video_musiq"
    aliases = ["video_musiq"]
