import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from ...annotations import ProcessedMeasurementInputType
from ...util import get_device, get_device_and_dtype_from_module, logger
from ..base import Measurement


class LearnedPerceptualImagePatchSimilarity(Measurement):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) - a deep learning based
    perceptual metric that uses features from pre-trained neural networks.
    Lower values indicate more similar images (0 = identical).
    """

    media_type = ("image", "image")
    name = "learned_perceptual_image_patch_similarity"
    aliases = ["lpips"]

    def setup(self) -> None:
        """Initialize the LPIPS model."""
        # Initialize LPIPS model with AlexNet backbone (most commonly used)
        # net_type='alex' is faster, 'vgg' is more accurate but slower
        self.lpips_model = LPIPS(net_type="alex", reduction="mean")

        # Move to device
        self.lpips_model = self.lpips_model.to(get_device())

        # Set to evaluation mode
        self.lpips_model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A tuple of two images as [3, h, w] float32 tensors in range [0, 1].
        :return: The LPIPS distance (lower = more similar, 0 = identical).
        """
        (reference, hypothesis) = input

        # LPIPS expects 4D tensors (batch dimension)
        # Add batch dimension if not present
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        if hypothesis.ndim == 3:
            hypothesis = hypothesis.unsqueeze(0)

        # Move to same device as model
        device, dtype = get_device_and_dtype_from_module(self.lpips_model)
        reference = reference.to(device, dtype=dtype)
        hypothesis = hypothesis.to(device, dtype=dtype)

        # Compute LPIPS
        # Note: torchmetrics LPIPS expects inputs in [0, 1] range
        with torch.no_grad():
            distance = self.lpips_model(reference, hypothesis)

        # Return scalar value
        return distance.item()
