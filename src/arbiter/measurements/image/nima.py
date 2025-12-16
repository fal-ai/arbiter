import torch

from ...annotations import ProcessedMeasurementInputType
from ...util import get_device, get_device_and_dtype_from_module, logger
from ..base import Measurement


class NIMA(Measurement):
    """
    Neural Image Assessment (NIMA) - aesthetic quality assessment.
    Uses pyiqa's implementation. Higher is better; typically normalized to [0, 1].
    """

    media_type = ("image",)
    name = "nima"
    aliases = ["nima"]

    def setup(self) -> None:
        """Initialize the NIMA model via pyiqa."""
        try:
            import pyiqa  # type: ignore
        except Exception as e:  # pragma: no cover - dependency hint
            raise ImportError(
                f"NIMA requires the 'pyiqa' package. Install with: pip install pyiqa\nException: {e}"
            ) from e

        # Create metric; downloads weights on first use
        self._pyiqa = pyiqa
        self.nima_model = pyiqa.create_metric("nima", device=get_device())
        logger.debug("Initialized NIMA metric via pyiqa")

        # pyiqa metrics are nn.Modules; ensure eval mode
        if hasattr(self.nima_model, "eval"):
            self.nima_model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A single image as [C, H, W] float32 tensor in [0, 1].
        :return: The NIMA aesthetic score (higher = better; often in [0, 1]).
        """
        (image,) = input

        # Ensure 4D tensor (batch dimension)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Move to same device as model
        device, dtype = get_device_and_dtype_from_module(self.nima_model)
        image = image.to(device, dtype=dtype)

        with torch.no_grad():
            score = self.nima_model(image)

        return float(score.item())
