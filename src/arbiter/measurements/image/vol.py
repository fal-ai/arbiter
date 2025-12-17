import torch
import torch.nn.functional as F

from ...annotations import ProcessedMeasurementInputType
from ..base import Measurement


class VarianceOfLaplacian(Measurement):
    """
    Variance of Laplacian (VoL) sharpness measure.
    Higher values typically indicate a sharper image.

    Values are scaled to match cv2.Laplacian(image, cv2.CV_64F).var() on uint8 [0, 255] images.
    """

    media_type = ("image",)
    name = "variance_of_laplacian"
    aliases = ["vol", "lapvar"]

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A single image as [C, H, W] float32 tensor in [0, 1].
        :return: Variance of Laplacian (higher = sharper).
        """
        (image,) = input

        if image.ndim != 3:
            raise ValueError("Expected image tensor of shape [C, H, W]")

        # Convert to grayscale [1, H, W] in [0, 255] range to match cv2.Laplacian().var()
        if image.shape[0] == 3:
            r, g, b = image[0:1], image[1:2], image[2 : 2 + 1]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        elif image.shape[0] == 1:
            gray = image
        else:
            # Fallback: average across channels
            gray = image.mean(dim=0, keepdim=True)

        # Scale to [0, 255] to produce values comparable to cv2.Laplacian().var()
        gray = gray * 255.0

        # Add batch dimension -> [1,1,H,W]
        gray_b = gray.unsqueeze(0)

        # Laplacian kernel (discrete 4-neighborhood)
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=gray_b.dtype,
            device=gray_b.device,
        ).view(1, 1, 3, 3)

        lap = F.conv2d(gray_b, kernel, padding=1)
        lap = lap.squeeze(0).squeeze(0)

        var = torch.var(lap, unbiased=False)
        return float(var.item())
