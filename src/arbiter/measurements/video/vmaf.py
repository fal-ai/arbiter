import torch
from torchmetrics.video import VideoMultiMethodAssessmentFusion as VMAF

from ...annotations import ProcessedMeasurementInputType
from ...util import get_device, get_device_and_dtype_from_module, logger
from ..base import Measurement


class VideoMultiMethodAssessmentFusion(Measurement):
    """
    Video Multi-Method Assessment Fusion (VMAF) between two videos.
    VMAF is a perceptual video quality assessment algorithm that combines multiple
    quality metrics into a single score. Returns a value typically between 0-100,
    where 100 means identical videos.
    """

    media_type = ("video", "video")
    name = "video_multi_method_assessment_fusion"
    aliases = ["vmaf"]

    def setup(self) -> None:
        """Initialize the VMAF model."""
        # Initialize VMAF model
        # VMAF typically uses default parameters for most use cases
        self.vmaf_model = VMAF()

        # Move to device
        self.vmaf_model = self.vmaf_model.to(get_device())

        # Set to evaluation mode
        self.vmaf_model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> list[float]:
        """
        :param input: A tuple of two videos as [f, c, h, w] float32 tensors in range [0, 1].
        :return: The VMAF score (typically 0-100, 100 = identical).
        """
        (reference, hypothesis) = input

        # VMAF expects 5D tensors (batch, channels, frames, height, width)
        reference = reference.unsqueeze(0).permute(0, 2, 1, 3, 4)
        hypothesis = hypothesis.unsqueeze(0).permute(0, 2, 1, 3, 4)

        # Move to same device as model
        device, dtype = get_device_and_dtype_from_module(self.vmaf_model)
        reference = reference.to(device, dtype=dtype)
        hypothesis = hypothesis.to(device, dtype=dtype)

        # Compute VMAF
        with torch.no_grad():
            vmaf_scores = self.vmaf_model(reference, hypothesis)[0]

        return vmaf_scores.tolist()
