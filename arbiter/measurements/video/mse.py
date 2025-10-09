from ..image.mse import ImageMeanSquaredError
from .image import VideoImageComparisonMeasurement


class VideoMeanSquaredError(VideoImageComparisonMeasurement):
    """
    A measurement that uses MSE to compare two videos.
    """

    image_cls = ImageMeanSquaredError
    name = "video_mean_squared_error"
    aliases = ["video_mse"]
