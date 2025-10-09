from collections import OrderedDict

import torch
from torchmetrics.multimodal.clip_iqa import (
    CLIPImageQualityAssessment as TMCLIPImageQualityAssessment,
)

from ...annotations import ProcessedMeasurementInputType
from ...util import logger
from ..base import Measurement


class CLIPIQA(Measurement):
    """
    CLIP-IQA: No-reference image quality assessment using CLIP.
    Accepts only an image. Uses statically-configured prompt pairs and returns a dict of scores per key.
    custom_prompts maps a key to a tuple of (positive_prompt, negative_prompt).
    """

    media_type = ("image",)
    name = "clip_image_quality_assessment"
    aliases = ["clip_iqa"]

    # Mapping name -> (positive_prompt, negative_prompt)
    custom_prompts: OrderedDict[str, tuple[str, str]] | None = None
    prompt_keys: list[str] = []

    def setup(self) -> None:
        """Initialize models for each configured prompt pair."""
        try:
            # Default prompts if none provided
            if self.custom_prompts is None:
                self.custom_prompts = OrderedDict(
                    [
                        ("quality", ("a low quality photo", "a high quality photo")),
                    ]
                )
            self.prompt_keys = list(self.custom_prompts.keys())

            self._build_model()
        except Exception as e:  # pragma: no cover - dependency hint
            raise ImportError(
                f"CLIP-IQA requires optional dependencies. Install with: pip install transformers timm piq\nException: {e}"
            ) from e

    def set_custom_prompts(
        self, prompts: dict[str, tuple[str, str]] | OrderedDict[str, tuple[str, str]]
    ) -> None:
        """Set custom prompt pairs and rebuild internal models."""
        self.custom_prompts = (
            prompts if isinstance(prompts, OrderedDict) else OrderedDict(prompts)
        )
        self.prompt_keys = list(self.custom_prompts.keys())
        self._build_model()

    def _build_model(self) -> None:
        """Build the CLIP-IQA model."""
        self._model = TMCLIPImageQualityAssessment(
            prompts=tuple(self.custom_prompts.values())
        )
        if torch.cuda.is_available():
            logger.debug(f"Moving CLIP-IQA model to GPU")
            self._model = self._model.cuda()
        self._model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> dict[str, float]:
        """
        :param input: A tuple with a single image [C, H, W] in [0, 1].
        :return: Dict mapping prompt-pair keys to their CLIP-IQA scores (higher = better).
        """
        (image,) = input

        # Ensure 4D image tensor (batch dimension)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        device = getattr(self._model, "device", next(self._model.parameters()).device)
        img = image.to(device)

        with torch.no_grad():
            scores = self._model(img)

        if len(self.prompt_keys) == 1:
            results = {self.prompt_keys[0]: torch.max(scores).item()}
        else:
            results = {
                key: torch.max(scores[f"user_defined_{i}"]).item()
                for i, key in enumerate(self.prompt_keys)
            }

        return results
