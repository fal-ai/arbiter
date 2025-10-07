import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from ...annotations import ProcessedMeasurementInputType
from ...util import logger
from ..base import Measurement


class StructuralSimilarityIndexMeasure(Measurement):
    """
    Structural Similarity Index Measure (SSIM) between two images.
    SSIM is a perceptual metric that considers changes in structural information,
    luminance, and contrast. Returns a value between -1 and 1, where 1 means identical images.
    """

    media_type = ("image", "image")
    name = "structural_similarity_index_measure"
    aliases = ["ssim"]

    def setup(self) -> None:
        """Initialize the SSIM model."""
        if SSIM is None:
            raise ImportError(
                "SSIM requires the 'torchmetrics' package. "
                "Install it with: pip install torchmetrics"
            )

        # Initialize SSIM with default parameters
        # data_range=1.0 for images in [0, 1] range
        self.ssim_model = SSIM(data_range=1.0)

        # Move to GPU if available
        if torch.cuda.is_available():
            logger.debug("Moving SSIM model to GPU")
            self.ssim_model = self.ssim_model.cuda()

        # Set to evaluation mode
        self.ssim_model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A tuple of two images as [3, h, w] float32 tensors in range [0, 1].
        :return: The SSIM value between -1 and 1 (1 = identical).
        """
        (reference, hypothesis) = input

        # SSIM expects 4D tensors (batch dimension)
        # Add batch dimension if not present
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        if hypothesis.ndim == 3:
            hypothesis = hypothesis.unsqueeze(0)

        # Move to same device as metric
        device = self.ssim_model.device
        reference = reference.to(device)
        hypothesis = hypothesis.to(device)

        # Compute SSIM
        with torch.no_grad():
            ssim_value = self.ssim_model(reference, hypothesis)

        # Return scalar value
        return ssim_value.item()
