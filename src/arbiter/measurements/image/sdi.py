import torch
from torchmetrics.functional.image.d_lambda import spectral_distortion_index as tm_sdi

from ...annotations import ProcessedMeasurementInputType
from ..base import Measurement


class SpectralDistortionIndex(Measurement):
    """
    Spectral Distortion Index (D_lambda) - measures spectral distortion between two images.
    Lower values indicate more similar spectra (0 = identical).
    """

    media_type = ("image", "image")
    name = "spectral_distortion_index"
    aliases = ["d_lambda", "sdi"]

    def setup(self) -> None:
        """No stateful setup required; use functional API to avoid buffers."""

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A tuple of two images as [C, H, W] float32 tensors in range [0, 1].
        :return: The Spectral Distortion Index (lower = better, 0 = identical).
        """
        (reference, hypothesis) = input

        # Metric expects 4D tensors (batch dimension)
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        if hypothesis.ndim == 3:
            hypothesis = hypothesis.unsqueeze(0)

        # Compute SDI using stateless functional API on CPU to minimize VRAM usage
        # Ensure tensors are on CPU and detached
        reference_cpu = reference.detach().to("cpu")
        hypothesis_cpu = hypothesis.detach().to("cpu")

        with torch.no_grad():
            value = tm_sdi(reference_cpu, hypothesis_cpu, reduction="elementwise_mean")

        return float(value.item())
