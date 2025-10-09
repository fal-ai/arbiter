from ..image.vol import VarianceOfLaplacian
from .image import VideoImageMeasurement


class VideoVarianceOfLaplacian(VideoImageMeasurement):
    """
    A measurement that uses Variance of Laplacian to measure a video.
    """

    image_cls = VarianceOfLaplacian
    name = "video_variance_of_laplacian"
    aliases = ["video_vol", "video_lapvar"]
