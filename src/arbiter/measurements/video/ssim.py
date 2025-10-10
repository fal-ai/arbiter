from ..image.ssim import StructuralSimilarityIndexMeasure
from .image import VideoImageComparisonMeasurement


class VideoStructuralSimilarityIndexMeasure(VideoImageComparisonMeasurement):
    """
    A measurement that uses SSIM to compare two videos.
    """

    image_cls = StructuralSimilarityIndexMeasure
    name = "video_structural_similarity_index_measure"
    aliases = ["video_ssim"]
