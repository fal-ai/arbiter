import torch
from torchmetrics.multimodal.clip_score import CLIPScore as TMCLIPScore

from ...annotations import ProcessedMeasurementInputType
from ...util import get_device, get_device_and_dtype_from_module, logger
from ..base import Measurement


class CLIPScore(Measurement):
    """
    CLIPScore: Measures similarity between text and image using CLIP.
    Accepts (text, image) and returns a similarity score (typically 0-100).
    """

    media_type = ("text", "image")
    name = "clip_score"
    aliases = ["clipscore", "clip_score"]

    def setup(self) -> None:
        """Initialize CLIPScore metric (downloads model weights on first use)."""
        try:
            self.clip_score = TMCLIPScore()
        except Exception as e:  # pragma: no cover - dependency hint
            raise ImportError(
                f"CLIPScore requires optional dependencies. Install with: pip install transformers timm\nException: {e}"
            ) from e

        # Move to device
        self.clip_score = self.clip_score.to(get_device())
        self.clip_score.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A tuple of (text, image) where text is str and image is [C, H, W] in [0, 1].
        :return: CLIPScore similarity (higher = better).
        """
        (text_prompt, image) = input

        # Ensure 4D image tensor (batch dimension)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Move to same device as model
        device, dtype = get_device_and_dtype_from_module(self.clip_score)
        image = image.to(device, dtype=dtype)

        with torch.no_grad():
            score = self.clip_score(image, text_prompt)

        return float(score.item())
