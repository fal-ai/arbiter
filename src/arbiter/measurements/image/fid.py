import torch
from torchmetrics.image import FrechetInceptionDistance as FID

from ...annotations import ProcessedMeasurementInputType
from ...util import get_device, get_device_and_dtype_from_module, logger
from ..base import Measurement


class FrechetInceptionDistance(Measurement):
    """
    Fréchet Inception Distance (FID) - measures the distance between
    feature distributions of real and generated images using the Inception network.
    Lower values indicate more similar distributions.

    This is an aggregate measurement that compares distributions of images,
    not individual image pairs.
    """

    media_type = ("image", "image")
    name = "frechet_inception_distance"
    aliases = ["fid"]
    aggregate = True  # This is an aggregate measurement

    def setup(self) -> None:
        """Initialize the FID model."""
        # Initialize FID with default parameters
        # feature=2048 uses the final average pooling features
        # normalize=True normalizes the input images
        self.fid_model = FID(feature=2048, normalize=True)

        # Move to device - we cannot use float64 on MPS, so we have to use cpu instead
        device = get_device()
        if device.type != "mps":
            self.fid_model = self.fid_model.to(device)

        # Set to evaluation mode
        self.fid_model.eval()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> float:
        """
        :param input: A list of tuples, where each tuple contains (real_image, generated_image).
                     Images should be [3, h, w] float32 tensors in range [0, 1].
        :return: The FID score (lower = more similar distributions).
        """
        # Reset the internal state
        self.fid_model.reset()
        device, dtype = get_device_and_dtype_from_module(self.fid_model)

        # Process each pair
        for real_image, generated_image in input:
            # FID expects 4D tensors (batch dimension)
            if real_image.ndim == 3:
                real_image = real_image.unsqueeze(0)
            if generated_image.ndim == 3:
                generated_image = generated_image.unsqueeze(0)

            # Convert to uint8 if needed (FID expects uint8 images)
            if real_image.dtype is torch.float32:
                real_image = (real_image * 255).to(torch.uint8)
            if generated_image.dtype is torch.float32:
                generated_image = (generated_image * 255).to(torch.uint8)

            # Move to same device as metric
            real_image = real_image.to(device, dtype=dtype)
            generated_image = generated_image.to(device, dtype=dtype)

            # Update FID with this pair
            # real=True for real images, real=False for generated images
            self.fid_model.update(real_image, real=True)
            self.fid_model.update(generated_image, real=False)

        # Compute final FID score
        with torch.no_grad():
            fid_score = self.fid_model.compute()

        # Return scalar value
        return fid_score.item()
