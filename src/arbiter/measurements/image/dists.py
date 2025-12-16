import torch
from torchmetrics.image import DeepImageStructureAndTextureSimilarity as DISTS

from ...annotations import ProcessedMeasurementInputType
from ...util import get_device, get_device_and_dtype_from_module, logger
from ..base import Measurement


class DeepImageStructureAndTextureSimilarity(Measurement):
    """
    Deep Image Structure and Texture Similarity (DISTS) - a perceptual metric
    that uses deep features from VGG to measure both structure and texture similarity.
    Lower values indicate more similar images (0 = identical).
    """

    media_type = ("image", "image")
    name = "deep_image_structure_and_texture_similarity"
    aliases = ["dists"]

    def setup(self) -> None:
        """Initialize the DISTS model."""
        # Initialize DISTS with default parameters
        # reduction='mean' averages the score across the batch
        self.dists_model = DISTS(reduction="mean")
        # Move to device
        self.dists_model = self.dists_model.to(get_device())
        # Set to evaluation mode
        self.dists_model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A tuple of two images as [3, h, w] float32 tensors in range [0, 1].
        :return: The DISTS distance (lower = more similar, 0 = identical).
        """
        (reference, hypothesis) = input

        # DISTS expects 4D tensors (batch dimension)
        # Add batch dimension if not present
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        if hypothesis.ndim == 3:
            hypothesis = hypothesis.unsqueeze(0)

        # Move to same device as metric
        device, dtype = get_device_and_dtype_from_module(self.dists_model)
        reference = reference.to(device, dtype=dtype)
        hypothesis = hypothesis.to(device, dtype=dtype)

        # Compute DISTS
        with torch.no_grad():
            distance = self.dists_model(reference, hypothesis)

        # Return scalar value
        return distance.item()
