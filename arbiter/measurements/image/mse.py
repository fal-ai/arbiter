import torch

from ...annotations import ProcessedMeasurementInputType
from ..base import Measurement


class ImageMeanSquaredError(Measurement):
    """
    Mean Squared Error (MSE) between two images.
    Measures the average squared difference between corresponding pixels.
    """

    media_type = ("image", "image")
    name = "image_mean_squared_error"
    aliases = ["mse"]

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A tuple of two images as [3, h, w] float32 tensors in range [0, 1].
        :return: The mean squared error between the images.
        """
        (reference, hypothesis) = input

        # Calculate MSE: mean of squared differences
        mse = torch.mean((reference - hypothesis) ** 2)
        return mse.item()
