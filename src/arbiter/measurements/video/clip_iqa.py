from ..image.clip_iqa import CLIPIQA
from .image import VideoImageMeasurement


class VideoCLIPIQA(VideoImageMeasurement):
    """
    A measurement that uses CLIPIQA to measure a video.
    """

    image_cls = CLIPIQA
    name = "video_clip_image_quality_assessment"
    aliases = ["video_clip_iqa"]
