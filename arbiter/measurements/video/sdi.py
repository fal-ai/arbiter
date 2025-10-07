from ..image.sdi import SpectralDistortionIndex
from .image import VideoImageComparisonMeasurement


class VideoSpectralDistortionIndex(VideoImageComparisonMeasurement):
    """
    A measurement that uses Spectral Distortion Index to compare two videos.
    """

    image_cls = SpectralDistortionIndex
    name = "video_spectral_distortion_index"
    aliases = ["video_d_lambda", "video_sdi"]
