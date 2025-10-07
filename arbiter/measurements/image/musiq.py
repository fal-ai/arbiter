import torch

from ...annotations import ProcessedMeasurementInputType
from ...util import logger
from ..base import Measurement


class MUSIQ(Measurement):
    """
    MUSIQ - Multi-scale Image Quality transformer.
    Uses pyiqa's implementation. Higher is better; scale depends on model but often normalized.
    """

    media_type = ("image",)
    name = "musiq"
    aliases = ["musiq"]

    def setup(self) -> None:
        """Initialize the MUSIQ model via pyiqa."""
        try:
            import pyiqa  # type: ignore
        except Exception as e:  # pragma: no cover - dependency hint
            raise ImportError(
                f"MUSIQ requires the 'pyiqa' package. Install with: pip install pyiqa\nException: {e}"
            ) from e

        # Create metric; downloads weights on first use
        self._pyiqa = pyiqa
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.musiq_model = pyiqa.create_metric("musiq", device=device)
        logger.debug("Initialized MUSIQ metric via pyiqa")

        # Ensure eval mode
        if hasattr(self.musiq_model, "eval"):
            self.musiq_model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A single image as [C, H, W] float32 tensor in [0, 1].
        :return: The MUSIQ quality score (higher = better).
        """
        (image,) = input

        # Ensure 4D tensor (batch dimension)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        with torch.no_grad():
            score = self.musiq_model(image)

        return float(score.item())
