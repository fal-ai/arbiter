from ..image.clip_score import CLIPScore
from .image import VideoImageTextAlignmentMeasurement


class VideoCLIPScore(VideoImageTextAlignmentMeasurement):
    """
    A measurement that uses CLIPScore to measure video-text alignment.
    """

    image_cls = CLIPScore
    name = "video_clip_score"
    aliases = ["video_clipscore"]
