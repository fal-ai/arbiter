from ..image.dists import DeepImageStructureAndTextureSimilarity
from .image import VideoImageComparisonMeasurement


class VideoDeepImageStructureAndTextureSimilarity(VideoImageComparisonMeasurement):
    """
    A measurement that uses DISTS to compare two videos.
    """

    image_cls = DeepImageStructureAndTextureSimilarity
    name = "video_deep_image_structure_and_texture_similarity"
    aliases = ["video_dists"]
