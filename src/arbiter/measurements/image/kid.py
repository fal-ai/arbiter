import torch
from torchmetrics.image import KernelInceptionDistance as KID

from ...annotations import ProcessedMeasurementInputType
from ...util import logger
from ..base import Measurement


class KernelInceptionDistance(Measurement):
    """
    Kernel Inception Distance (KID) - measures the dissimilarity between
    feature distributions of real and generated images using the Inception network.
    Lower values indicate more similar distributions.

    This is an aggregate measurement that compares distributions of images,
    not individual image pairs.
    """

    media_type = ("image", "image")
    name = "kernel_inception_distance"
    aliases = ["kid"]
    aggregate = True  # This is an aggregate measurement

    def setup(self) -> None:
        """Initialize the KID model."""
        # Initialize KID with default parameters
        # subset_size=50 is the default for computing polynomial kernel MMD
        self.kid_model = KID(subset_size=50)

        # Move to GPU if available
        if torch.cuda.is_available():
            logger.debug("Moving KID model to GPU")
            self.kid_model = self.kid_model.cuda()

    def __call__(  # type: ignore[override]
        self,
        input: ProcessedMeasurementInputType,
    ) -> dict[str, float]:
        """
        :param input: A list of tuples, where each tuple contains (real_image, generated_image).
                     Images should be [3, h, w] float32 tensors in range [0, 1].
        :return: Dictionary with KID mean and std values.
        """
        # Reset the internal state
        self.kid_model.reset()
        self.kid_model.subset_size = min(len(input), 1000)

        # Process each pair
        for real_image, generated_image in input:
            # KID expects 4D tensors (batch dimension)
            if real_image.ndim == 3:
                real_image = real_image.unsqueeze(0)
            if generated_image.ndim == 3:
                generated_image = generated_image.unsqueeze(0)

            if real_image.dtype is torch.float32:
                real_image = (real_image * 255).to(torch.uint8)
            if generated_image.dtype is torch.float32:
                generated_image = (generated_image * 255).to(torch.uint8)

            # Move to same device as metric
            device = self.kid_model.device
            real_image = real_image.to(device)
            generated_image = generated_image.to(device)

            # Update KID with this pair
            # real=True for real images, real=False for generated images
            self.kid_model.update(real_image, real=True)
            self.kid_model.update(generated_image, real=False)

        # Compute final KID score
        with torch.no_grad():
            kid_mean, kid_std = self.kid_model.compute()

        return kid_mean.item()
