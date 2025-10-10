from ..image.lpips import LearnedPerceptualImagePatchSimilarity
from .image import VideoImageComparisonMeasurement


class VideoLearnedPerceptualImagePatchSimilarity(VideoImageComparisonMeasurement):
    """
    A measurement that uses LPIPS to compare two videos.
    """

    image_cls = LearnedPerceptualImagePatchSimilarity
    name = "video_learned_perceptual_image_patch_similarity"
    aliases = ["video_lpips"]
