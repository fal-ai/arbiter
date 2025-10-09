import torch
from torchmetrics.image.arniqa import ARNIQA as TMARNIQA

from ...annotations import ProcessedMeasurementInputType
from ...util import logger
from ..base import Measurement


class ARNIQA(Measurement):
    """
    ARNIQA: No-reference image quality assessment.
    Returns a quality score where higher is better. When normalize=True, scores are in [0, 1].
    """

    media_type = ("image",)
    name = "arniqa"
    aliases = ["arniqa"]

    def setup(self) -> None:
        """Initialize the ARNIQA model."""
        # Use default regressor trained on KonIQ-10k and normalized output in [0, 1]
        try:
            self.arniqa_model = TMARNIQA(regressor_dataset="koniq10k", normalize=True)
        except Exception as e:  # pragma: no cover - dependency hint
            # Provide helpful message for optional deps (e.g., timm/einops)
            raise ImportError(
                "ARNIQA requires optional dependencies. Install with: pip install timm einops"
            ) from e

        # Move to GPU if available
        if torch.cuda.is_available():
            logger.debug("Moving ARNIQA model to GPU")
            self.arniqa_model = self.arniqa_model.cuda()

        # Set to evaluation mode
        self.arniqa_model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A tuple with one image as [C, H, W] float32 tensor in range [0, 1].
        :return: The ARNIQA quality score (higher = better, in [0, 1] when normalized).
        """
        (image,) = input

        # Ensure 4D tensor (batch dimension)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Move to same device as model
        device = getattr(
            self.arniqa_model, "device", next(self.arniqa_model.parameters()).device
        )
        image = image.to(device)

        # Compute ARNIQA
        with torch.no_grad():
            score = self.arniqa_model(image)

        # Return scalar value
        return float(score.item())
